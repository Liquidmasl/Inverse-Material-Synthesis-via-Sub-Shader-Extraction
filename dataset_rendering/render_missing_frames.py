import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
import sys

rendered_images_dir = r'F:\Dropbox\Masl Stuff\BachelorArbeit\Datasets\FixedDataset24\Frames'
blend_file_path = r'BiggerShaderDatasetjune24.blend'
start_frame = 0
end_frame = 25000  # Adjust based on your project
batch_size = 100  # Number of frames to render in each batch
max_workers = 4  # Adjust based on your system's capabilities

def find_missing_frames(directory, start, end):
    rendered_frames = set()
    frame_pattern = re.compile(r'(\d+).exr')
    for filename in os.listdir(directory):
        match = frame_pattern.search(filename)
        if match:
            frame_number = int(match.group(1))
            rendered_frames.add(frame_number)
    all_frames = set(range(start, end + 1))
    missing_frames = all_frames - rendered_frames
    return sorted(missing_frames)

def render_frames(batch):
    start, end = batch
    try:
        subprocess.run(['blender', '-b', blend_file_path, '-o', f"{rendered_images_dir}\\", '-s', str(start), '-e', str(end), '-a'], check=True)
    except subprocess.CalledProcessError:
        print(f"Blender crashed while rendering frames {start} to {end}. Retrying...", file=sys.stderr)
        render_frames(batch)  # Retry rendering the batch

def main():
    missing_frames = find_missing_frames(rendered_images_dir, start_frame, end_frame)


    # find contiguous batches of missing frames
    batches = []
    current_batch = None
    for frame in missing_frames:
        if current_batch is None:
            current_batch = [frame, frame]
        elif frame == current_batch[1] + 1:
            current_batch[1] = frame
        else:
            batches.append(tuple(current_batch))
            current_batch = [frame, frame]
    batches.append(tuple(current_batch))
    current_batch = [frame, frame]

    print(f"Found {len(missing_frames)} missing frames. Rendering {len(batches)} batches with {max_workers} workers.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(render_frames, batches)

if __name__ == '__main__':
    main()