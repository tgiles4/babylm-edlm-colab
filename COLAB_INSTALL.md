# Installation Guide for Google Colab

This guide provides step-by-step instructions for setting up the Energy-Diffusion-LLM codebase in Google Colab.

## Prerequisites

1. **Google Colab with A100 GPU** (40GB recommended)
2. **Google Drive** - for storing tokenizer, checkpoints, and cached data
3. **BabyLM datasets** - located at `/content/drive/MyDrive/babylm-edlm/data/train_10M` and `/content/drive/MyDrive/babylm-edlm/data/train_100M`

## Step-by-Step Setup

### Step 1: Upload Codebase to Colab

1. Zip your `Energy-Diffusion-LLM` repository
2. In Colab, upload the zip file:
   ```python
   from google.colab import files
   uploaded = files.upload()  # Upload your zip file
   ```
3. Extract the codebase:
   ```python
   !unzip -q Energy-Diffusion-LLM.zip -d /content/
   !cd /content/Energy-Diffusion-LLM
   ```

### Step 2: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3: Install Dependencies

Choose one of the following methods:

#### Method 0: Using Conda with requirements.yaml (Advanced)

If you prefer to use conda (the original environment file format):

```python
# Option A: Using the conda installation script
!cd /content/Energy-Diffusion-LLM && bash install_colab_conda.sh

# Option B: Using the Python conda installation script
exec(open('/content/Energy-Diffusion-LLM/install_colab_conda.py').read())

# Option C: Manual conda installation
# Install Miniconda first
!wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
!bash miniconda.sh -b -p /usr/local/miniconda
!rm miniconda.sh

# Add to PATH
import os
os.environ['PATH'] = '/usr/local/miniconda/bin:' + os.environ.get('PATH', '')

# Create environment (note: Colab already has PyTorch, so we'll skip it)
!conda env create -f /content/Energy-Diffusion-LLM/requirements.yaml -n edlm

# To use the environment:
import sys
sys.path.insert(0, '/usr/local/miniconda/envs/edlm/lib/python3.9/site-packages')
```

**Note**: Conda installation is more complex in Colab because:
- Colab already has PyTorch/CUDA installed, which may conflict with conda's versions
- You need to manage Python paths manually
- **Recommendation**: Use pip installation (Method A, B, or C below) for simplicity

#### Method A: Using the pip installation script (Recommended)

#### Method A: Using the installation script (Easiest)

```python
!cd /content/Energy-Diffusion-LLM && bash install_colab.sh
```

#### Method B: Using the Python installation script

```python
exec(open('/content/Energy-Diffusion-LLM/install_colab.py').read())
```

#### Method C: Manual pip installation (step-by-step)

**Important**: Installation order matters! Install in this order:

```python
# Step 1: Install ninja (required build tool)
!pip install -q ninja

# Step 2: Install base requirements
!pip install -q -r /content/Energy-Diffusion-LLM/requirements.txt

# Step 3: Install flash-attention (REQUIRED, may take 5-10 minutes)
!pip install -q flash-attn --no-build-isolation

# Step 4: Install causal-conv1d (REQUIRED for DiMamba models)
!pip install -q causal-conv1d

# Step 5: Install mamba-ssm (REQUIRED for DiMamba models, may take a few minutes)
!pip install -q mamba-ssm
```

**Note**: All of these packages are REQUIRED because they are imported in `models/__init__.py` and the model files:
- `flash-attn`: Required by `models/dit.py` and `models/autoregressive.py`
- `causal-conv1d`: Required by `models/dimamba.py`
- `mamba-ssm`: Required by `models/dimamba.py`

### Step 4: Verify Installation

```python
import torch
import lightning as L
import transformers
import datasets

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
print(f"Lightning version: {L.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"Datasets version: {datasets.__version__}")
```

Expected output:
- PyTorch: 2.x.x
- CUDA available: True
- CUDA version: 12.1 or similar
- Lightning: 2.2.1
- Transformers: 4.38.2
- Datasets: 2.18.0

### Step 5: Prepare Google Drive Structure

Create the necessary directories in Google Drive:

```python
import os

# Create directories in Google Drive
drive_base = "/content/drive/MyDrive/babylm-edlm"
os.makedirs(f"{drive_base}/cache", exist_ok=True)
os.makedirs(f"{drive_base}/checkpoints", exist_ok=True)
os.makedirs(f"{drive_base}/outputs", exist_ok=True)

print("Google Drive directories created!")
```

### Step 6: Create/Train Tokenizer

Since you're training from scratch, you need to create a tokenizer first:

```python
# Train a BPE tokenizer from your BabyLM dataset
!python /content/Energy-Diffusion-LLM/train_tokenizer.py \
    --data_dir /content/drive/MyDrive/babylm-edlm/data/train_10M \
    --output_path /content/drive/MyDrive/babylm-edlm/tokenizer/ \
    --vocab_size 32000

# Verify tokenizer was created
tokenizer_path = "/content/drive/MyDrive/babylm-edlm/tokenizer/tokenizer.json"
if os.path.exists(tokenizer_path):
    print("✓ Tokenizer created successfully!")
    # Test it
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_file(tokenizer_path))
    print(f"  Vocab size: {tokenizer.vocab_size}")
else:
    print("✗ Tokenizer creation failed!")
```

**Note**:
- This will take 5-15 minutes depending on dataset size
- The tokenizer will be saved to Google Drive for persistence
- You only need to do this once - reuse the same tokenizer for all training runs

## Troubleshooting

### Issue: flash-attn installation fails

**Solution**: Try installing with specific CUDA version:
```python
# Make sure ninja is installed first
!pip install ninja
!pip install flash-attn --no-build-isolation --no-cache-dir
```

**Important**: flash-attn is REQUIRED (not optional) because it's imported in `models/dit.py` and `models/autoregressive.py`. The code will fail to import without it.

### Issue: causal-conv1d or mamba-ssm installation fails

**Solution**: These are also REQUIRED for DiMamba models. Try:
```python
# Make sure ninja is installed first
!pip install ninja
!pip install causal-conv1d --no-cache-dir
!pip install mamba-ssm --no-cache-dir
```

**Note**: If you're only using DIT models (not DiMamba), you technically don't need these, but they're still imported in the codebase, so you'll need them installed.

### Issue: CUDA out of memory

**Solutions**:
1. Reduce batch size in config: `python main.py loader.global_batch_size=256`
2. Use gradient accumulation: Increase `trainer.accumulate_grad_batches`
3. Use smaller model: `python main.py model=tiny`

### Issue: Import errors

**Solution**: Make sure you're in the correct directory:
```python
import sys
sys.path.insert(0, '/content/Energy-Diffusion-LLM')
```

### Issue: Google Drive mount fails

**Solution**: Re-run the mount command and authorize again:
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

## Quick Test

After installation, test that everything works:

```python
# Change to codebase directory
%cd /content/Energy-Diffusion-LLM

# Test import
import main
import dataloader
import diffusion
print("✓ All imports successful!")

# Test config loading
import hydra
from omegaconf import DictConfig
with hydra.initialize(config_path="configs"):
    cfg = hydra.compose(config_name="config", overrides=["data=babylm"])
    print("✓ Config loaded successfully!")
    print(f"  - Dataset: {cfg.data.train}")
    print(f"  - Use Energy: {cfg.use_energy}")
```

## Next Steps

Once installation is complete, proceed to training:

```bash
# Train DIT without Energy, 10M tokens
python main.py data=babylm use_energy=False dataset_size=10M noise=cosine

# Train DIT with Energy, 100M tokens
python main.py data=babylm use_energy=True dataset_size=100M noise=cosine data.data_dir=/content/drive/MyDrive/babylm-edlm/data/train_100M
```

See `BABYLM_COLAB_SETUP.md` for more details on configuration and usage.

