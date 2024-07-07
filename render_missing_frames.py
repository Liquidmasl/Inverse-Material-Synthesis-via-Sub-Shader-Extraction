import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
import sys

rendered_images_dir = r'F:\Dropbox\Masl Stuff\BachelorArbeit\Datasets\FixedDataset24\Frames'
blend_file_path = '/path/to/your/project.blend'
start_frame = 1
end_frame = 100000  # Adjust based on your project
batch_size = 1000  # Number of frames to render in each batch
max_workers = 4  # Adjust based on your system's capabilities

def find_missing_frames(directory, start, end):
    rendered_frames = set()
    frame_pattern = re.compile(r'frame(\d+)')
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
        subprocess.run(['blender', '-b', blend_file_path, '-s', str(start), '-e', str(end), '-a'], check=True)
    except subprocess.CalledProcessError:
        print(f"Blender crashed while rendering frames {start} to {end}. Retrying...", file=sys.stderr)
        render_frames(batch)  # Retry rendering the batch

def main():
    missing_frames = find_missing_frames(rendered_images_dir, start_frame, end_frame)
    batches = [(missing_frames[i], missing_frames[min(i + batch_size - 1, len(missing_frames) - 1)]) for i in range(0, len(missing_frames), batch_size)]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(render_frames, batches)

if __name__ == '__main__':
    main()