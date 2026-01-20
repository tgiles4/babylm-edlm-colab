# Training Guide: BabyLM Dataset on Colab with A100 40GB

This guide explains how to train a diffusion model on the BabyLM dataset using pretokenized .bin files in Google Colab with a single A100 40GB GPU.

## Required Files

To train the model, you need the following files:

### 1. Codebase Files
Upload or clone the entire Energy-Diffusion-LLM repository. The following files are essential:

**Core Python Files:**
- `main.py` - Main training entry point
- `dataloader.py` - Data loading utilities (modified to support .bin files)
- `dataloader_bin.py` - **NEW** Custom loader for pretokenized .bin files
- `deprecated/diffusion.py` - Diffusion model implementation (will be symlinked to `diffusion.py`)
- `utils.py` - Utility functions
- `noise_schedule.py` - Noise schedule implementation
- `models/` directory - Model architectures (dit.py, dimamba.py, etc.)

**Configuration Files:**
- `configs/config.yaml` - Main configuration
- `configs/data/babylm.yaml` - **NEW** BabyLM dataset configuration
- `configs/model/` - Model configurations (tiny.yaml, small.yaml, medium.yaml)
- `configs/noise/` - Noise schedule configurations
- `configs/strategy/` - Training strategy configurations
- `configs/lr_scheduler/` - Learning rate scheduler configurations
- `configs/callbacks/` - Callback configurations

### 2. Data Files
- **`data/` folder** containing 6 pretokenized .bin files:
  - These files should be created using `tokenize_corpus.py`
  - Each .bin file contains a list of PyTorch tensors (one tensor per document)
  - File naming: Can be any name ending in `.bin` (e.g., `file1_tokenized.bin`, `file2_tokenized.bin`, etc.)

### 3. Tokenizer File
- **`tokenizer.json`** - BPE tokenizer file (same one used to create the .bin files)
  - Created using `create_bpe_tokenizer.py` or similar
  - Must match the tokenizer used for tokenization

## File Structure

Your Colab workspace should have this structure:

```
/content/Energy-Diffusion-LLM/
├── data/                          # Data folder with .bin files
│   ├── file1_tokenized.bin
│   ├── file2_tokenized.bin
│   ├── file3_tokenized.bin
│   ├── file4_tokenized.bin
│   ├── file5_tokenized.bin
│   └── file6_tokenized.bin
├── tokenizer.json                 # BPE tokenizer
├── main.py
├── dataloader.py                  # Modified to support babylm-bin
├── dataloader_bin.py              # NEW: Custom .bin file loader
├── diffusion.py                   # Symlink to deprecated/diffusion.py
├── deprecated/
│   └── diffusion.py               # Actual diffusion implementation
├── configs/
│   ├── config.yaml
│   ├── data/
│   │   └── babylm.yaml            # NEW: BabyLM config
│   ├── model/
│   │   ├── tiny.yaml
│   │   ├── small.yaml
│   │   └── medium.yaml
│   └── ...
└── (other codebase files)
```

## Quick Start

1. **Open the Colab notebook**: `train_babylm_colab.ipynb`

2. **Follow the cells in order**:
   - Install dependencies
   - Upload codebase (zip file or git clone)
   - Upload data folder and tokenizer
   - Fix import paths
   - Verify setup
   - Configure training parameters
   - Start training

3. **Key Configuration Parameters** (adjust for A100 40GB):
   ```python
   config_overrides = {
       'data': 'babylm',
       'model': 'small',              # or 'tiny' for less memory
       'loader.global_batch_size': 256,
       'loader.batch_size': 256,
       'trainer.max_steps': 100000,
       'trainer.val_check_interval': 5000,
       'optim.lr': 3e-4,
   }
   ```

## Memory Considerations for A100 40GB

### Model Sizes:
- **tiny**: ~50M parameters, ~200MB memory
- **small**: ~125M parameters, ~500MB memory
- **medium**: ~350M parameters, ~1.4GB memory

### Batch Size Recommendations:
- **tiny model**: Can handle batch_size up to 512
- **small model**: Can handle batch_size up to 256
- **medium model**: Can handle batch_size up to 128

### Sequence Length:
- Default: 1024 tokens
- Reducing to 512 can save memory if needed

## Training Process

1. **Data Loading**: The `dataloader_bin.py` module:
   - Loads all .bin files from the `data/` directory
   - Concatenates all documents
   - Chunks them into sequences of `block_size` (default 1024)
   - Adds BOS and EOS tokens
   - Returns a HuggingFace Dataset

2. **Training**: Uses PyTorch Lightning for distributed training
   - Automatic mixed precision (bf16)
   - Gradient accumulation for effective larger batch sizes
   - EMA (Exponential Moving Average) for model weights
   - Checkpointing for resuming training

3. **Checkpoints**: Saved in `checkpoints/` directory
   - `last.ckpt` - Latest checkpoint (auto-resume from here)
   - Training automatically resumes if interrupted

## Troubleshooting

### Out of Memory (OOM)
- Reduce `global_batch_size` (try 128 or 64)
- Use `model: tiny` instead of `small`
- Reduce sequence length in model config

### Import Errors
- Ensure `diffusion.py` symlink exists (created in Step 4)
- Check all dependencies are installed
- Verify Python path includes the codebase directory

### Data Loading Errors
- Verify `data/` folder exists and contains .bin files
- Check tokenizer.json path is correct
- Ensure tokenizer matches the one used for tokenization
- Verify .bin files are valid PyTorch tensors

### CUDA Errors
- Restart Colab runtime: Runtime → Restart runtime
- Check CUDA version compatibility
- Verify A100 GPU is allocated (check in Step 5)

## Files Created/Modified

### New Files:
1. **`dataloader_bin.py`** - Custom dataset loader for .bin files
2. **`configs/data/babylm.yaml`** - BabyLM dataset configuration
3. **`train_babylm_colab.ipynb`** - Colab training notebook
4. **`TRAINING_GUIDE.md`** - This guide

### Modified Files:
1. **`dataloader.py`** - Added support for `babylm-bin` dataset name and .json tokenizer loading

## Next Steps

After training:
- Checkpoints are saved in `checkpoints/` directory
- Use the model for sampling/generation
- Evaluate on validation set
- Fine-tune hyperparameters as needed

For more details, see the Colab notebook: `train_babylm_colab.ipynb`






