import os
import random
import shutil

source_folder = "/home/faymek/MPCompress/data/coco2017/val2017"
destination_folder = "/home/faymek/MPCompress/data/COCO_2017_val_sel100/val"

num_files = 50

os.makedirs(destination_folder, exist_ok=True)

all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]


if len(all_files) < num_files:
    raise ValueError(f"Not enough files in source folder. Found {len(all_files)}, but need {num_files}.")


selected_files = random.sample(all_files, num_files)


for filename in selected_files:
    src_path = os.path.join(source_folder, filename)
    dst_path = os.path.join(destination_folder, filename)
    shutil.copy2(src_path, dst_path)

print(f"Successfully copied {len(selected_files)} files from '{source_folder}' to '{destination_folder}'.")
