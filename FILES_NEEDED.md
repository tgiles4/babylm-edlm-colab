# Files Needed for Training on Colab

## Summary

To train the diffusion model on BabyLM dataset in Colab with a single A100 40GB, you need:

### 1. **Codebase Files** (Upload entire repository)
All files from the Energy-Diffusion-LLM repository, including:
- Core Python files (`main.py`, `dataloader.py`, etc.)
- Configuration files in `configs/`
- Model files in `models/`
- **NEW files created for this task:**
  - `dataloader_bin.py` - Custom loader for .bin files
  - `configs/data/babylm.yaml` - BabyLM dataset config
  - `train_babylm_colab.ipynb` - Colab training notebook

### 2. **Data Files** (6 .bin files)
- Folder: `data/`
- Contents: 6 pretokenized .bin files created with `tokenize_corpus.py`
- Format: Each .bin file contains a list of PyTorch tensors (int16)

### 3. **Tokenizer File**
- File: `tokenizer.json`
- Format: BPE tokenizer JSON file (same one used for tokenization)
- Created with: `create_bpe_tokenizer.py` or similar

## Quick Checklist

Before starting training, ensure you have:

- [ ] Entire codebase uploaded/cloned to Colab
- [ ] `data/` folder with 6 .bin files
- [ ] `tokenizer.json` file (BPE tokenizer)
- [ ] All dependencies installed (handled in notebook Step 1)
- [ ] A100 GPU allocated in Colab

## File Structure in Colab

```
/content/Energy-Diffusion-LLM/
├── data/                          # Your 6 .bin files go here
│   ├── *.bin (6 files)
├── tokenizer.json                 # Your BPE tokenizer
├── main.py
├── dataloader.py
├── dataloader_bin.py              # NEW
├── diffusion.py                   # Symlink (created automatically)
├── configs/
│   ├── config.yaml
│   ├── data/
│   │   └── babylm.yaml            # NEW
│   └── ...
└── train_babylm_colab.ipynb       # NEW - Training notebook
```

## Next Steps

1. Open `train_babylm_colab.ipynb` in Colab
2. Follow the cells step by step
3. Upload your files when prompted
4. Start training!

For detailed instructions, see `TRAINING_GUIDE.md`.





