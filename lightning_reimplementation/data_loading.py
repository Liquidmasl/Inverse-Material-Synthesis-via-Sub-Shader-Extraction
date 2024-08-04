import math
import os

import numpy as np
import pandas as pd
import torch
import yaml
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from lightning_reimplementation.util import get_file_with_highest_number

class BlenderSuperShaderRendersDataset(Dataset):
    def __init__(self, param_dir, gram_directory, training_run_path, transform=None, add_key_colors_to_input=False,
                  input_size=-1, force_additional_importance_calc=False, importance_path=None, index_path=None, **kwargs):
        self.param_dir = param_dir
        self.gram_directory = gram_directory
        self.transform = transform
        self.add_key_colors_to_input = add_key_colors_to_input
        self.gram_path_cols = []
        self.dataset = None
        self.feature_mask = None
        self.training_run_path = training_run_path

        self.num_params = 41

        if index_path is not None and os.path.exists(index_path):
            self.dataset = pd.read_csv(index_path, header=[0, 1])
        else:
            self.init_dataframe()

        self.dataset.to_csv(os.path.join(self.training_run_path, 'index.csv'), index=False)

        self.local_importances_path = os.path.join(self.training_run_path, 'feature_importances')
        os.makedirs(self.local_importances_path, exist_ok=True)

        if input_size > 0:
            self.init_importances(input_size, force_additional_importance_calc, importance_path)

    def init_importances(self, take_top_n_features, force_more_importance_calc, importance_path):
        self.feature_mask = None

        self.feature_mask = self.get_feature_importance(force_more_importance_calc, importance_path)

        most_acc_importances_path = get_file_with_highest_number(self.local_importances_path, r"feature_importance_##.npy")
        feature_importance = np.load(most_acc_importances_path)

        sorted_indices = np.argsort(feature_importance)[::-1]

        top_indices = sorted_indices[:take_top_n_features]

        self.feature_mask = top_indices.copy()

    def init_dataframe(self):
        samples = set([x.split('_')[0] for x in os.listdir(self.gram_directory)])
        self.gram_path_cols = [f"gra_{i}_path" for i in range(5)]

        mult_idx_tuples = []

        rows = []
        for sample in tqdm(samples, desc="Loading samples into dataset"):
            row = pd.read_csv(os.path.join(self.param_dir, f"parameters_frame_{int(sample)}.txt"), delimiter=',',
                              header=0, dtype=np.float32)

            if mult_idx_tuples == []:
                column_names = row.columns
                column_categories = ['frame'] + ['parameters'] * self.num_params + ['key_colors'] * 9
                mult_idx_tuples = list(zip(column_categories, column_names)) + [('gram_paths', p_col) for p_col in
                                                                                self.gram_path_cols]

            for i in range(5):
                row[self.gram_path_cols[i]] = os.path.join(self.gram_directory, f"{sample}_{i}")

            rows.append(row)

        self.dataset = pd.concat(rows)
        self.dataset['frame'] = self.dataset['frame'].astype(int)
        multi_index = pd.MultiIndex.from_tuples(mult_idx_tuples)
        self.dataset.columns = multi_index




    def get_feature_importance(self, force_more_importance_calc, importances_path):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor

        prev = get_file_with_highest_number(importances_path, r"feature_importance_##.npy", return_num=True)

        if prev is not None:
            path, num = prev

            importances = np.load(path)

            np.save(os.path.join(self.local_importances_path, f"feature_importance_{num}.npy"), importances)
            if not force_more_importance_calc:
                return importances

            feature_importances = [importances] * num
            num_checked = num

        else:
            feature_importances = []
            num_checked = 0

        num_same = 0
        prev_means = None

        for data, targets in tqdm(DataLoader(self, batch_size=100, shuffle=True, num_workers=2), total=(len(self) // 100) + 1):

            model = MultiOutputRegressor(RandomForestRegressor(n_estimators=50, n_jobs=-1))
            model.fit(data.numpy(), targets.numpy())
            feature_importance = model.estimators_[0].feature_importances_
            feature_importances.append(feature_importance)
            num_checked += len(data)

            mean = np.mean(feature_importances, axis=0)
            np.save(os.path.join(self.local_importances_path, f"feature_importance_{num_checked}.npy"), mean)

            if prev_means is not None:
                diff = np.abs(prev_means - mean)
                print(np.mean(diff))
                if np.all(diff < 0.001):
                    num_same += 1
                else:
                    num_same = 0

            prev_means = mean

            if num_same > 3:
                break

            print(f"importances did barely change for the last {num_same} batches")

        return mean

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        dataset_row = self.dataset.iloc[index]
        input_grams = [torch.load(gram_filename) for gram_filename in dataset_row['gram_paths']]
        input_gram_diags = [gram[np.triu_indices(gram.shape[0], k=1)] for gram in input_grams]
        input_grams_concat = torch.cat(input_gram_diags)

        params = dataset_row['parameters'].values.astype(np.float32)

        if self.add_key_colors_to_input:
            params = np.concatenate((params, dataset_row['key_colors'].values))

        if self.feature_mask is not None:
            input_grams_concat = input_grams_concat[self.feature_mask]

        params = torch.tensor(params)

        return (input_grams_concat, params)


class BlenderShaderDataModule(LightningDataModule):
    def __init__(self, batch_size, training_run_path, configs, transform=None):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transform
        self.training_run_path = training_run_path
        self.configs = configs



    def setup(self, stage=None):
        # Load or initialize the dataset here

        self.dataset = BlenderSuperShaderRendersDataset(training_run_path = self.training_run_path, **self.configs)

        if 'splits_calc' in self.configs:
            self.splits = self.metadata['splits_calc']
        else:
            np_splits = np.array(self.configs['splits'])
            if (np_splits > 1).all() and np.issubdtype(np_splits.dtype, np.integer):

                train_start = 0
                train_end = train_start + np_splits[0]

                test_start = train_end
                test_end = test_start + np_splits[1]

                val_start = test_end
                val_end = test_end + np_splits[2]


            elif (abs(np_splits.sum() - 1) < 0.001):

                ds_len = len(self.dataset)

                train_start = 0
                train_end = math.floor(train_start + ds_len * np_splits[0])

                test_start = train_end
                test_end = math.floor(test_start + ds_len * np_splits[1])

                val_start = test_end
                val_end = math.floor(val_start + ds_len * np_splits[2])

            else:
                raise ValueError('Split definition is invalid, must be int lengths of splits, or fractions of 1 summing to 1')

            self.configs['splits_calc'] = {
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'val_start': val_start,
                'val_end': val_end
            }

            with open(os.path.join(self.training_run_path, 'configs_after_setup.yaml'), 'w') as file:
                yaml.dump(self.configs, file)



    def train_dataloader(self):
        subs = torch.utils.data.Subset(self.dataset, range(self.configs['splits_calc']['train_start'], self.configs['splits_calc']['train_end']))
        return DataLoader(subs, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        subs = torch.utils.data.Subset(self.dataset, range(self.configs['splits_calc']['val_start'], self.configs['splits_calc']['val_end']))
        return DataLoader(subs, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        subs = torch.utils.data.Subset(self.dataset, range(self.configs['splits_calc']['test_start'], self.configs['splits_calc']['test_end']))
        return DataLoader(subs, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def teardown(self, stage=None):
        # Clean up after training or testing
        ...

