# BabyLM Training Setup for Google Colab

This document describes the modifications made to support training BabyLM models in Google Colab with flexible configuration options.

## Quick Start - Installation

### Option 0: Using Conda with requirements.yaml (Advanced)

If you prefer to use the original `requirements.yaml` conda environment file:

```python
# Using the conda installation script
!cd /content/Energy-Diffusion-LLM && bash install_colab_conda.sh

# Or using Python script
exec(open('/content/Energy-Diffusion-LLM/install_colab_conda.py').read())
```

**Note**: Conda installation in Colab is more complex because Colab already has PyTorch/CUDA installed.
For simplicity, we recommend using pip installation (Option 1, 2, or 3 below).

### Option 1: Using the pip installation script (Recommended)

In a Colab cell, run:
```python
# Upload your codebase zip file first, then:
!cd /content/Energy-Diffusion-LLM && bash install_colab.sh
```

### Option 2: Using the Python installation script

In a Colab cell, run:
```python
exec(open('/content/Energy-Diffusion-LLM/install_colab.py').read())
```

### Option 3: Manual installation (step-by-step)

**Important**: Installation order matters! These packages are REQUIRED (not optional) because they're imported in the model files.

In a Colab cell, run:
```python
# Step 1: Install ninja (required build tool)
!pip install -q ninja

# Step 2: Install base requirements
!pip install -q -r /content/Energy-Diffusion-LLM/requirements.txt

# Step 3: Install flash-attention (REQUIRED - used by dit.py and autoregressive.py)
# This may take 5-10 minutes
!pip install -q flash-attn --no-build-isolation

# Step 4: Install causal-conv1d (REQUIRED - used by dimamba.py)
!pip install -q causal-conv1d

# Step 5: Install mamba-ssm (REQUIRED - used by dimamba.py)
# This may take a few minutes
!pip install -q mamba-ssm

# Verify installation
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Verify required imports
import flash_attn
import causal_conv1d
import mamba_ssm
print("✓ All required dependencies imported successfully!")
```

**Note**:
- The `requirements.yaml` file is a Conda environment file. For Colab, use `requirements.txt` instead.
- All dependencies (flash-attn, causal-conv1d, mamba-ssm) are REQUIRED because they're imported in `models/__init__.py` and the model files.
- Installation may take 10-15 minutes total due to compilation of CUDA extensions.

## Summary of Changes

### 1. BabyLM Dataset Loading (`dataloader.py`)
- **Added `get_babylm_dataset()` function**: Reads all `.train` files from a directory, combines them, and automatically splits into 90% train / 10% validation
- **Updated `get_dataset()` function**: Added support for `babylm` dataset name with `data_dir` parameter
- **Updated `get_dataloaders()` function**: Passes `data_dir` from config to dataset loader

### 2. Configuration Files

#### `configs/data/babylm.yaml`
- Configured to load BabyLM `.train` files
- Default paths for Colab environment:
  - `data_dir`: `/content/drive/MyDrive/babylm-edlm/data/train_10M` (change to `train_100M` for 100M tokens)
  - `tokenizer_name_or_path`: `/content/drive/MyDrive/babylm-edlm/tokenizer/tokenizer.json`
  - `cache_dir`: `/content/drive/MyDrive/babylm-edlm/cache`

#### `configs/config.yaml`
- **Added `use_energy` flag**: `False` for standard DIT, `True` for DIT with Energy (EBM)
- **Added `dataset_size` flag**: `10M` or `100M` (for logging/naming)
- **Updated Google Drive paths**:
  - Outputs: `/content/drive/MyDrive/babylm-edlm/outputs/...`
  - Checkpoints: `/content/drive/MyDrive/babylm-edlm/checkpoints`

#### `configs/noise/cosine.yaml` (NEW)
- Cosine noise schedule configuration

### 3. Model Instantiation (`main.py`)
- **Updated `_train()` function**: Conditionally instantiates `Diffusion` or `EBM` based on `use_energy` flag
- **Updated `_load_from_checkpoint()` function**: Handles both model types when loading checkpoints

## Usage Examples

### Train DIT without Energy, 10M tokens, cosine noise schedule:
```bash
python main.py data=babylm use_energy=False dataset_size=10M noise=cosine
```

### Train DIT with Energy, 100M tokens, cosine noise schedule:
```bash
python main.py data=babylm use_energy=True dataset_size=100M noise=cosine data.data_dir=/content/drive/MyDrive/babylm-edlm/data/train_100M
```

### Train DIT without Energy, 10M tokens (default):
```bash
python main.py data=babylm noise=cosine
```

## File Structure in Colab

```
/content/
├── Energy-Diffusion-LLM/          # Your codebase (uploaded as zip)
│   ├── main.py
│   ├── dataloader.py
│   ├── diffusion.py
│   ├── configs/
│   │   ├── config.yaml
│   │   ├── data/
│   │   │   └── babylm.yaml
│   │   └── noise/
│   │       └── cosine.yaml
│   └── ...
├── drive/
│   └── MyDrive/
│       └── babylm-edlm/            # Google Drive resources
│           ├── tokenizer.json      # Your BPE tokenizer
│           ├── cache/              # Cached tokenized datasets (auto-created)
│           ├── checkpoints/        # Model checkpoints (auto-created)
│           └── outputs/            # Training outputs/logs (auto-created)
└── data/                           # BabyLM datasets (mounted or copied)
    ├── train_10M/
    │   ├── bnc_spoken.train
    │   ├── childes.train
    │   ├── gutenberg.train
    │   ├── open_subtitles.train
    │   ├── simple_wiki.train
    │   └── switchboard.train
    └── train_100M/                  # Same 6 files, larger size
        ├── bnc_spoken.train
        └── ...
```

## Key Features

1. **Automatic 90/10 Split**: No separate validation set needed - automatically splits training data
2. **Automatic Caching**: Tokenized datasets are cached after first run for fast subsequent loads
3. **Flexible Model Selection**: Easy switching between DIT and DIT+Energy via config
4. **Flexible Dataset Size**: Easy switching between 10M and 100M tokens via config
5. **Google Drive Persistence**: All outputs, checkpoints, and cached data saved to Google Drive

## Configuration Override Examples

### Change dataset size:
```bash
python main.py data.data_dir=/content/drive/MyDrive/babylm-edlm/data/train_100M dataset_size=100M
```

### Change tokenizer path:
```bash
python main.py data.tokenizer_name_or_path=/path/to/tokenizer.json
```

### Change cache directory:
```bash
python main.py data.cache_dir=/path/to/cache
```

### Change checkpoint directory:
```bash
python main.py checkpointing.save_dir=/path/to/checkpoints
```

## Creating the Tokenizer

Since you're training from scratch, you need to create a BPE tokenizer first:

```python
# Train tokenizer from BabyLM dataset
!python train_tokenizer.py \
    --data_dir /content/drive/MyDrive/babylm-edlm/data/train_10M \
    --output_path /content/drive/MyDrive/babylm-edlm/tokenizer/ \
    --vocab_size 32000
```

**Note**: The script will automatically create `tokenizer.json` inside the specified directory.

This will:
- Read all `.train` files from the specified directory
- Train a BPE tokenizer with the specified vocabulary size
- Save it as `tokenizer.json` to Google Drive
- Take 5-15 minutes depending on dataset size

**Important**:
- Do this BEFORE starting training
- You only need to do this once - reuse the same tokenizer for all experiments
- The tokenizer must match the one used during training

## Notes

- First run will tokenize the dataset (takes 5-60 minutes depending on size)
- Subsequent runs load from cache instantly
- All cached data is saved to Google Drive for persistence across Colab sessions
- The 90/10 split uses a fixed seed (42) for reproducibility

