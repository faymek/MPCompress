import os
import shutil
# sel 100 files from coco2017 val2017, from the COCO_val2017_sel100 dataset
base_dir = "/home/faymek/MPCompress/"
src_dir = "/path/to/dataset/coco2017/val2017"
dst_dir = f"{base_dir}/data/dataset/COCO_val2017_sel100/img"
txt_file = f"{base_dir}/data/dataset/COCO_val2017_sel100/COCO_val2017_sel100.txt"

with open(txt_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

img_paths = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    parts = line.split(" ", 1)
    filename, caption = parts
    filename = filename.split("_")[-1]

    img_path = os.path.join(src_dir, filename)
    dst_img_path = os.path.join(dst_dir, filename)
    if os.path.exists(img_path):
        img_paths.append(img_path)
        shutil.copy(img_path, dst_img_path)
    else:
        print(f"⚠️ File not found: {img_path}")
print(f"{len(img_paths)} files copied from {src_dir} to {dst_dir}")