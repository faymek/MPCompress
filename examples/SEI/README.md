# README

Extracting visual information from downstream task networks, to enhance LLM task understanding

## Python Envoriment

There is a version confilct with `transformers` with Poetry. So please manaully install these dependencies using pip.

```
pip install transformers==4.51.3 accelerate
```

You may do following to load models from HuggingFace.
```
export HF_ENDPOINT=https://hf-mirror.com 
```

## Demo
The demo/ folder currently contains two example Jupyter notebooks:

- `object_detection.ipynb`:
  - Demonstrates how to preprocess images to generate SEI (Scene Encoding Information).
  Currently, it performs object detection as the preprocessing step, and will be extended to support more vision tasks in the future.

- `video_understand_with_SEI.ipynb`:
  - Shows a comparison of the LLM's response quality when provided with SEI information versus when no SEI is given, using the same input image.

## Pipline
The end-to-end workflow is as follows:：

1. **Data Sampling**
Run scripts/00_prepare_dataset.py to randomly select 50 images from the coco/val2017 dataset as inference examples.

2. **SEI Generation**
Run scripts/01_object_detection.py to generate SEI (JSON files) for each selected image.

3. **Interaction with LLM**
Run scripts/02_video_understand_with_SEI.py to feed the generated SEI into the LLM, enabling enhanced video understanding or Q&A tasks.

## Planned Extensions
- Support for additional types of SEI (e.g., scene segmentation, relationship graphs).

- More downstream benchmarks to quantify the impact of SEI on LLM performance.