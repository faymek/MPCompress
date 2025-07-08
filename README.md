# Introduction

This project is the official implementation of the Multi-Purpose Compression test platform.


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

# Link to local CompressAI source code.
poetry run pip install --editable .

```

## Install MMCV

Install `mmcv==2.1.0` and `mmsegmentation==1.2.1`. mmcv needs to be built from source. Note that mmcv 2.0 has many apis that differs from mmcv 1.0. For detailed instructions, see: https://mmcv.readthedocs.io/en/latest/get_started/installation.html

Build mmcv from source.
```sh
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v2.1.0

pip install -r requirements/optional.txt
nvcc --version
gcc --version

# missing steps in mmcv docs
python setup.py build_ext
python setup.py develop

# Validate the installation
python .dev_scripts/check_installation.py
```

Install mmsegmentation using mim
```sh
pip install -U openmim
mim install "mmsegmentation==1.2.1"
```


# Dataset and Model Preparation

We provide packaged datasets and pretrained model checkpoints at the following link:  
[Insert download link here]

Alternatively, you may download them manually by following the instructions below:  
[Add download instructions if applicable]


# Testing Instructions

## Activate Environment
Activate the Poetry environment:
```sh
poetry shell
```

## Testing FCM-LM

Implementation for the paper:  
"Feature Coding in the Era of Large Models: Dataset, Test Conditions, and Benchmark"

*Note: This test script is adapted from [FCM-LM](https://github.com/chansongoal/FCM-LM)*

### Test Options:

1. **DINOv2 Classification Model with VTM Compression**
```sh
CUDA_VISIBLE_DEVICES=0 python examples/fcm-lm/run_dinov2-cls_vtm_cls.py
```

2. **Hyperprior Model with Hyperprior Compression**
```sh
CUDA_VISIBLE_DEVICES=0 python mpcompress/eval/run_dinov2_cls_hyperprior.py
```




# Acknowledgement

Special thanks to Donghui Feng, Fengxi Zhang, Bo Gao and Zekai Liu, for their valuable contributions in building this test platform.

Special thanks to Yifan Ma, Qiaoxi Chen, and Yenan Xu, for their valuable contributions in building FCM-LM.