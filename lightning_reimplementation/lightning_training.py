import os

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import StepLR

import wandb
import yaml
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


from lightning_reimplementation.data_loading import BlenderShaderDataModule

torch.set_float32_matmul_precision('medium')

class MyModel(pl.LightningModule):
    def __init__(self, input_size, num_classes, loss_fns, loss_fn_name, optimiser_setup):
        super().__init__()
        self.loss_fns = loss_fns
        self.loss_fn_name = loss_fn_name
        self.optimiser_setup = optimiser_setup
        self.num_classes = num_classes

        self.layers = nn.ModuleList([
            nn.Linear(input_size, 2500),
            nn.Linear(2500, 2000),
            nn.Linear(2000, 1000),
            nn.Linear(1000, 1000),
            nn.Linear(1000, 1000),
            nn.Linear(1000, 1000),
            nn.Linear(1000, 1000),
            nn.Linear(1000, 1000),
            nn.Linear(1000, 1000),
            nn.Linear(1000, 500),
            nn.Linear(500, 500),
            nn.Linear(500, 500),
            nn.Linear(500, 100),
            nn.Linear(100, 100),
            nn.Linear(100, 100),
            nn.Linear(100, 100),
            nn.Linear(100, 50),
            nn.Linear(50, 50),
            nn.Linear(50, 50),
            nn.Linear(50, num_classes)
        ])

        self.apply(self.init_weights)

        self.validation_abs_errors = []

    def init_weights(self, m):
        if isinstance(m, nn.Linear):

            if m.out_features == self.num_classes:
                # Maybe use a different initialization for the final layer
                init.xavier_uniform_(m.weight)
            else:
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Apply ReLU to all but the last layer
                x = F.relu(x)
        x = torch.sigmoid(x)  # Apply sigmoid activation at the end
        return x

    def training_step(self, batch, batch_idx):
        data, targets = batch
        prediction = self(data)

        for name, func in self.loss_fns.items():
            loss = func(prediction, targets)
            self.log(f"train_{name}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, reduce_fx='mean')

            if name == self.loss_fn_name:
                ret = loss

        return ret

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        prediction = self(data)
        abs_error = torch.abs(prediction - targets)
        self.validation_abs_errors.append(abs_error)

        for name, func in self.loss_fns.items():
            self.log(f"val_{name}_loss", func(prediction, targets), on_step=True, on_epoch=True, prog_bar=True, logger=True, reduce_fx='mean')


    def on_validation_epoch_end(self):

        all_abs_errors = torch.cat(self.validation_abs_errors, dim=0)
        # Calculate the mean across all batches
        mean_abs_error = all_abs_errors.mean()
        self.log('val_mean_accuracy', 1 - mean_abs_error, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', lr, on_epoch=True)
        self.validation_abs_errors = []

    def on_epoch_end(self):
        super().on_epoch_end()



    def configure_optimizers(self):
        if self.optimiser_setup['optimiser'] == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.optimiser_setup['lr'])
        else:
            raise ValueError("Only adam is supported as of now")

        if self.optimiser_setup['scheduler'] == 'stepLR':
            scheduler = StepLR(optimizer, step_size=self.optimiser_setup['step_size'], gamma=self.optimiser_setup['gamma'])

        if self.optimiser_setup['use_scheduler']:
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': self.optimiser_setup['interval'],  # or 'step' for step-wise updates
                    'frequency': self.optimiser_setup['frequency'],  # how often the scheduler is applied
                    'monitor': self.optimiser_setup['monitor'],  # for schedulers like ReduceLROnPlateau
                }
            }
        else:
            return optimizer


class MSLELoss(nn.Module):
    def __init__(self, log_part):
        super().__init__()
        self.mse = nn.MSELoss()
        self.a = log_part

    def forward(self, pred, actual):
        ret = self.mse(pred, actual) + self.a * (self.mse(torch.log(pred + 0.0000001), torch.log(actual + 0.0000001)))
        return ret


# This is complete bs, i thought i can implement a loss method that takes the importance weights into account. but the importance weights are for the inputs , not the outputs. so all this overengineering was for nothing
#
# class Weighted_MSELoss(nn.Module):
#
#     def __init__(self, weights, interpolation_map):
#         super().__init__()
#         self.weights = torch.tensor(weights)
#         self.initial_weights = torch.tensor(weights)
#         self.interp_map = interpolation_map
#         self.weight_strength = -1
#         self.step(0)
#
#     def forward(self, pred, actual):
#         squared_errors = (pred - actual) ** 2
#         weighted_squared_errors = squared_errors * self.weights
#         return weighted_squared_errors.mean()
#
#     def interpolate_weight(self, epoch):
#         # Sort keys and extract boundary epochs and their corresponding weights
#         sorted_epochs = sorted(self.interp_map.keys())
#         weights = [self.interp_map[e] for e in sorted_epochs]
#
#         # Find the interval where the current epoch falls
#         for i in range(len(sorted_epochs)):
#             if sorted_epochs[i] <= epoch <= sorted_epochs[i + 1] :
#                 # Perform linear interpolation between the weights
#                 weight_start = weights[i]
#                 weight_end = weights[i + 1]
#                 epoch_start = sorted_epochs[i]
#                 epoch_end = sorted_epochs[i + 1]
#
#                 # Calculate the interpolated weight
#                 interpolated_weight = weight_start + (weight_end - weight_start) * (epoch - epoch_start) / (epoch_end - epoch_start)
#                 return interpolated_weight
#
#         return weights[-1]
#
#     def step(self, epoch):
#         new_weight_strength = self.interpolate_weight(epoch)
#
#         if new_weight_strength == self.weight_strength:
#             return
#
#         self.weight_strength = new_weight_strength
#
#         new_weights = self.weight_strength * self.initial_weights + (self.weight_strength - 1) * torch.ones_like(self.initial_weights)
#
#         if not torch.all(torch.isclose(self.weights, new_weights, atol=1e-4)):
#             self.weights[:] = new_weights



def start_training_run():

    config_path = r'configs.yaml'
    configs = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    wandb_logger = WandbLogger(project='2024Lightning', save_code=True, log_model='all')
    wandb_logger.experiment.config.update(configs)
    wandb_logger.experiment.log_code(".")
    run_folder = wandb_logger.experiment.dir

    data_module = BlenderShaderDataModule(
        batch_size=configs['batch_size'],
        configs=configs,
        training_run_path=run_folder,
        transform=None,
    )
    data_module.setup(stage='fit')


    loss_functions = {
        'mse': F.mse_loss,
        'msle': MSLELoss(configs.get('msle_log_part', 15)),
        # 'wmse': Weighted_MSELoss(weights, configs.get('wmse_imterpolation_map'))
    }

    model = MyModel(input_size=configs['input_size'], num_classes=41, loss_fn_name=configs['loss_fn'], loss_fns=loss_functions, optimiser_setup=configs['optimiser_setup'])
    wandb_logger.experiment.watch(model, 'all')


    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=configs['max_epochs'],
        accelerator='gpu',
        callbacks=[ModelCheckpoint(dirpath=os.path.join(run_folder, 'Models'), save_top_k=1, monitor=f"val_{configs['loss_fn']}_loss", mode='min')],
        log_every_n_steps=16
    )





    wandb.save(config_path)

    trainer.fit(model, data_module)



if __name__ == '__main__':

    start_training_run()

