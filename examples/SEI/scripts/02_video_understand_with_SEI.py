import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info
import json
from tqdm import tqdm

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
model = model.to("cuda")  # 显式将模型移动到CUDA设备

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")


# Preparation for inference
def infer_qwen_vl(messages):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text


def response_no_SEI(img_path):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_path,
                },
                {
                    "type": "text",
                    "text": "Describe the relative positions of the objects in the image precisely, without any vague statements.",
                },
            ],
        }
    ]

    response = infer_qwen_vl(messages)
    # print(response)
    return response


def respondse_with_SEI(img_path, SEI_dir):
    img_name = os.path.basename(img_path).split(".")[0]
    SEI_json = os.path.join(SEI_dir, f"SEI_{img_name}.json")
    with open(SEI_json, "r") as f:
        SEI_data = json.load(f)

    [width, height] = SEI_data["img_size"]
    print(width, height)
    positions_str = ""
    for obj_name, entry in SEI_data["Object Detection"].items():
        positions_str += f"- {obj_name}: {entry['bbox']}\n"
    print(positions_str)
    messages_SEI = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_path,
                },
                {
                    "type": "text",
                    "text": (
                        f"The image size is {width} x {height}.\n\n"
                        f"Here is the detected object position data in the image:\n"
                        "Each bounding box is in the format [x_min, y_min, x_max, y_max].\n"
                        f"{positions_str}\n"
                        "Please provide a detailed spatial description of these objects. "
                        "Describe their approximate locations within the image (for example, near the top left, center, bottom right), "
                        "their relative sizes, and how they are positioned with respect to each other."
                    ),
                },
            ],
        }
    ]

    response_SEI = infer_qwen_vl(messages_SEI)
    # print(response_SEI)
    return response_SEI


if __name__ == "__main__":

    data_dir = "/home/faymek/MPCompress/data"
    work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_dir = os.path.join(data_dir, "dataset", "ImageNet_val_sel100", "img")
    SEI_dir = os.path.join(work_dir, "results", "per-image-SEI")
    output_file = os.path.join(work_dir, "results", "all_response.jsonl")

    img_files = [
        f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    img_files.sort()

    with open(output_file, "w") as f_out:
        for img_file in tqdm(img_files):
            img_path = os.path.join(img_dir, img_file)

            output = response_no_SEI(img_path)
            output_SEI = respondse_with_SEI(img_path, SEI_dir)

            result = {
                "image": img_file,
                "response_no_SEI": output,
                "response_with_SEI": output_SEI,
            }

            f_out.write(json.dumps(result) + "\n")
