import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*no-op.*")
from collections import defaultdict
import json
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained("/home/gabriel/checkponits/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("/home/gabriel/checkponits/detr-resnet-50", revision="no_timm").to(device)
# model.eval()

def infer_img_detection(img_path, rlt_dir):
    image = Image.open(img_path)
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        SEI_dict = {}
        seen_objects = defaultdict(int)
        object_detection = {}
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            object_name = model.config.id2label[label.item()]
            seen_objects[object_name] += 1
            numbered_object_name = f"{object_name}_{seen_objects[object_name]}"
            box = [round(i, 2) for i in box.tolist()]
            object_detection[numbered_object_name] = {
                'bbox': box,
                'confidence_score': round(score.item(), 3)
            }
        # print(object_detection)
        SEI_dict['Object Detection'] = object_detection
        width, height = image.size
        SEI_dict['img_size'] = [width, height]
        # print(SEI_dict)


        image_name = os.path.basename(img_path).split('.')[0]
        with open(f'{rlt_dir}/SEI_{image_name}.json', 'w') as json_file:
            json.dump(SEI_dict, json_file, indent=4)


        del inputs, outputs, target_sizes, results
        torch.cuda.empty_cache()
        


img_dir = "/home/gabriel/SEI_work/data/img_test"
rlt_dir = "/home/gabriel/SEI_work/rlts/SEI"
img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
for img_file in tqdm(img_files):
    img_path = os.path.join(img_dir, img_file)
    infer_img_detection(img_path, rlt_dir)

# infer_img_detection(img_path, rlt_dir)