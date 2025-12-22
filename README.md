MPCompress is a PyTorch library and evaluation platform for multi-purpose compression research, providing a public testing environment for evaluating feature coding methods.

MPCompress currently provides:

* Multi-Purpose Compression (MPC) framework - a coding architecture designed to prioritize machine vision while retaining compatibility with human visual perception
* Feature Coding for Large Models (LaMoFC) framework - a feature coding framework for distributed large model deployments
* evaluation scripts and test platforms for comparing compression methods

## Installation

MPCompress supports Python 3.10+, PyTorch 2.4.0 and CUDA 12.1.

### Using Poetry (Recommended)

Poetry helps manage version-pinned virtual environments. First, install [Poetry](https://python-poetry.org/docs/#installation):

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Then, create the virtual environment and install the required Python packages:

```bash
cd MPCompress

# Install Python packages to new virtual environment.
poetry install
echo "Virtual environment created in $(poetry env list --full-path)"

# Link to local MPCompress source code.
poetry run pip install --editable .
```

## Documentation

* [Documentation](https://faymek.github.io/MPCompress)

## Dataset and Weights Preparation

We provide publicly available datasets and weights via the following link:

Share content: MPCompress-share
Link: https://pan.sjtu.edu.cn/web/share/2f9f14e05fa73c8742994aae67198dff
Extraction code: 1127

Please refer to the examples to download the needed resources and extract them into the current directory. The resulting directory structure should look like this:

```
MPCompress/
├─ data/
│   ├─ ADEChallengeData2016/
│   ├─ ImageNet_val_sel2k/
│   └─ VOC2012/
├─ weights/
│   ├─ dinov2/
│   └─ MPC/
```

The full directory structure is organized as follows:
```
MPCompress/
├─ data/         # dataset libraries
├─ features/     # extracted features
├─ runs/         # training logs and checkpoints
├─ models/       # vision model libraries, reserved for future use
├─ weights/      # pre-trained weights and released checkpoints
```

## Usage

Coding examples can be found in the `examples/` directory.

### Testing LaMoFC

> "Feature Coding in the Era of Large Models: Dataset, Test Conditions, and Benchmark"

Large models are often partitioned and deployed across multiple devices. In such distributed setups, intermediate features must be encoded and transmitted between nodes. LaMoFC is a feature coding framework designed to minimize the required bitrate under a specified task accuracy constraint, or conversely, to maximize task accuracy under a given bitrate limit.

For implementation details and usage examples, please refer to the directory `examples/lamofc/` and its dedicated [README](examples/lamofc/README.md).

This test script is adapted from the original [LaMoFC repository](https://github.com/chansongoal/LaMoFC). Note that results may vary slightly depending on the versions of Python libraries used.

### Testing MPC

The Multi-Purpose Compression (MPC) framework is a coding architecture designed to prioritize machine vision while retaining compatibility with human visual perception. It extracts general-purpose visual features at the encoder and enables low-complexity decoding of task-relevant information at the decoder. The framework adopts a multi-branch structure to support on-demand bitstream extraction, making it adaptable to a variety of downstream tasks.

For implementation details and usage examples, please refer to the directory `examples/mpc/` and its dedicated [README](examples/mpc/README.md).


## Acknowledgement

Special thanks to Donghui Feng, Bo Gao, Qingyue Ling, Fengxi Zhang and Zekai Liu, for their valuable contributions in building this test platform.

Special thanks to Yifan Ma, Qiaoxi Chen, and Yenan Xu, for their valuable contributions in building LaMoFC.

## Related Links

* [CompressAI repository](https://github.com/microsoft/CompressAI)
* [LaMoFC repository](https://github.com/chansongoal/LaMoFC)
* [DINOv2 repository](https://github.com/facebookresearch/dinov2)
* [VTM repository](https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM)
