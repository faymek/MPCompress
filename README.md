# Introduction

This project is the official implementation of the 


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

# Test

Activate Poetry environment
```sh
poetry shell
```

Test dinov2_cls model and VTM compression
```sh
python mpcompress/eval/run_dinov2_cls_vtm.py
```

Test hyperprior model and hyperprior compression
```sh
python mpcompress/eval/run_dinov2_cls_hyperprior.py
```

