import os

import yaml
from tqdm import tqdm
import torch


def normalise_grams(input_folder_path, output_folder_path, not_normalised_metadata_path, normalised_metadata_path, reset=False):

    with open(not_normalised_metadata_path, 'r') as file:
        input_metadata = yaml.load(file, Loader=yaml.FullLoader)

    # check if output metadata exists, if not create it
    try:
        with open(normalised_metadata_path, 'r') as file:
            output_metadata = yaml.load(file, Loader=yaml.FullLoader)
    except FileNotFoundError:
        output_metadata = {'mins': input_metadata['mins'], 'maxs': input_metadata['maxs'], 'processed_files': [set(), set(), set(), set(), set()]}

    if reset:
        output_metadata = {'mins': input_metadata['mins'], 'maxs': input_metadata['maxs'],
                           'processed_files': [set(), set(), set(), set(), set()]}

    os.makedirs(output_folder_path, exist_ok=True)

    for i in range(5):

        # if the already normalised files were processed with different mins and maxs, reprocess them
        # this can be seperated into the different layers to reduce the amount of files to process
        if output_metadata['mins'][i] != input_metadata['mins'][i] or output_metadata['maxs'][i] != input_metadata['maxs'][i]:
            print(f"Reprocessing layer {i} because min or max changed")
            output_metadata['processed_files'][i] = {}

    # join all processed sub sets to get a set of all normalised files
    all_processed_files = set()
    for set_ in output_metadata['processed_files']:
        all_processed_files.update(set_)

    # get all files in the input folder
    all_files = os.listdir(input_folder_path)
    all_unprocessed_files = set(all_files) - all_processed_files

    mins = input_metadata['mins']
    maxs = input_metadata['maxs']

    processed = 0
    # process all unprocessed files
    for file in tqdm(all_unprocessed_files):

        layer = int(file.split('_')[1])
        gram = torch.load(os.path.join(input_folder_path, file))

        gram = gram.subtract(mins[layer])
        gram = gram.div(maxs[layer] - mins[layer])

        if torch.max(gram) > 1 or torch.min(gram) < 0:
            print("ALAAARRMM")

        torch.save(gram, os.path.join(output_folder_path, file))
        output_metadata['processed_files'][layer].add(file)

        processed += 1
        if processed % 100 == 0:
            with open(normalised_metadata_path, 'w') as file:
                yaml.dump(output_metadata, file)


if __name__ == '__main__':

    normalise_grams(
        input_folder_path=r'/app/data/grams/not_normalised',
        output_folder_path=r'/app/data/grams/normalised',
        not_normalised_metadata_path=r'/app/data/grams/not_normalised_metadata.yaml',
        normalised_metadata_path=r'/app/data/grams/normalised_metadata.yaml'
    )




