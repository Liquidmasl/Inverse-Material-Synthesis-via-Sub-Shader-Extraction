import os

import pandas as pd
from tqdm import tqdm
import OpenEXR as exr

def quantize(value, n):
    # Calculate the size of each part
    part_size = 1.0 / n
    # Find which part the value falls into
    part = int(value / part_size)
    # Calculate the center of the part
    center = (part * part_size) + (part_size / 2)
    # Ensure the center value does not exceed 1
    return min(center, 1)

# Load parmater label data


# columns = ['frame'] + list(range(0, 52))
# df = pd.read_csv('/app/data/Labels.csv', header=None, names=columns)

multiplexing_columns = {4: 8,
17: 8,
26: 6,
35: 6,
36: 6,
37: 6,
38: 6,
39: 6,
40: 6}

loadDirectory = r'/app/data/renders'

for filename in tqdm(os.listdir(loadDirectory)):
    if filename.endswith('.exr'):
        exrfile = exr.InputFile(r"/app/data/renders/manyVars_0003.exr")
        header = exrfile.header()

        # Extract the notes attribute from the header
        frame = int(header['Frame'])
        notes = header['Note']



for column, n in multiplexing_columns.items():
    df[column] = df[column].apply(lambda x: quantize(x, n))

df.to_csv('/app/data/Labels_quantized.csv', index=False)

print('waiit')