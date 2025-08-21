# Introduction

This project is the official implementation of the Multi-Purpose Compression (MPC) test platform. It provides a public testing environment for evaluating feature coding methods.

# Environments Set Up

## Using poetry
Poetry helps manage version-pinned virtual environments. First, install [Poetry](https://python-poetry.org/docs/#installation):

```sh
curl -sSL https://install.python-poetry.org | python3 -
```

Then, create the virtual environment and install the required Python packages:

```sh
cd MPCompress

# Install Python packages to new virtual environment.
poetry install
echo "Virtual environment created in $(poetry env list --full-path)"

# Link to local MPCompress source code.
poetry run pip install --editable .

```

# Dataset and Model Preparation

We provide publicly available datasets and pre-trained backbone/head checkpoints via the following link:

- Link: https://pan.sjtu.edu.cn/web/share/ca76721e0799fc8411ff44c1c4cbf5f3  
- Access code: mkhr

Please download the resources and extract them into the current directory. The resulting directory structure should look like this:

```
data
├── dataset
│   ├── COCO_val2017_sel100
│   ├── ImageNet_val_sel100
│   ├── ImageNet_val_sel2k
│   ├── VOC2012
├── models
│   ├── backbone
│   ├── dinov2_cls_head
│   └── dinov2_seg_head
```


# Testing Instructions

## Activate Environment
Activate the Poetry environment:
```sh
poetry shell
```

## Testing FCM-LM

> "Feature Coding in the Era of Large Models: Dataset, Test Conditions, and Benchmark"

Large models are often partitioned and deployed across multiple devices. In such distributed setups, intermediate features must be encoded and transmitted between nodes. FCM-LM is a feature coding framework designed to minimize the required bitrate under a specified task accuracy constraint, or conversely, to maximize task accuracy under a given bitrate limit.

For implementation details and usage examples, please refer to the directory `examples/fcm-lm/` and its dedicated [README](examples/fcm-lm/README.md).

This test script is adapted from the original [FCM-LM repository](https://github.com/chansongoal/LaMoFC). Note that results may vary slightly depending on the versions of Python libraries used.

## Testing MPC

The Multi-Purpose Compression (MPC) framework is a coding architecture designed to prioritize machine vision while retaining compatibility with human visual perception. It extracts general-purpose visual features at the encoder and enables low-complexity decoding of task-relevant information at the decoder. The framework adopts a multi-branch structure to support on-demand bitstream extraction, making it adaptable to a variety of downstream tasks.

For implementation details and usage examples, please refer to the directory `examples/mpc/` and its dedicated [README](examples/mpc/README.md).

# Acknowledgement

Special thanks to Donghui Feng, Bo Gao, Qingyue Ling, Fengxi Zhang and Zekai Liu, for their valuable contributions in building this test platform.

Special thanks to Yifan Ma, Qiaoxi Chen, and Yenan Xu, for their valuable contributions in building FCM-LM.