# Tokenizer Training Guide

Since you're training from scratch, you need to create a BPE tokenizer from your BabyLM dataset before starting training.

## Quick Start

### Step 1: Train the Tokenizer

In a Colab cell, run:

```python
!python /content/Energy-Diffusion-LLM/train_tokenizer.py \
    --data_dir /content/drive/MyDrive/babylm-edlm/data/train_10M \
    --output_path /content/drive/MyDrive/babylm-edlm/tokenizer/ \
    --vocab_size 32000
```

**Note**: You can provide either a directory path (will create `tokenizer.json` inside) or a full file path ending in `.json`.

**Parameters:**
- `--data_dir`: Directory containing your `.train` files (e.g., `/content/drive/MyDrive/babylm-edlm/data/train_10M` or `/content/drive/MyDrive/babylm-edlm/data/train_100M`)
- `--output_path`: Where to save the tokenizer (should be in Google Drive for persistence)
- `--vocab_size`: Vocabulary size (default: 32000, common values: 16000, 32000, 50000)

### Step 2: Verify Tokenizer

```python
import os
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer

tokenizer_path = "/content/drive/MyDrive/babylm-edlm/tokenizer/tokenizer.json"
if os.path.exists(tokenizer_path):
    print("✓ Tokenizer found!")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_file(tokenizer_path))
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Test it
    test_text = "This is a test sentence."
    tokens = tokenizer.encode(test_text)
    print(f"  Test: '{test_text}' -> {tokens} -> '{tokenizer.decode(tokens)}'")
else:
    print("✗ Tokenizer not found!")
```

### Step 3: Start Training

Once the tokenizer is created, you can start training:

```bash
python main.py data=babylm noise=cosine use_energy=False
```

## What the Tokenizer Does

The `train_tokenizer.py` script:

1. **Reads all `.train` files** from the specified directory
2. **Trains a BPE (Byte Pair Encoding) tokenizer** using the HuggingFace `tokenizers` library
3. **Creates special tokens**: `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`, `[BOS]`, `[EOS]`
4. **Saves as `tokenizer.json`** in the format expected by the codebase

## Tokenizer Configuration

The tokenizer uses:
- **Normalization**: Lowercase, strip accents (NFD)
- **Pre-tokenization**: Whitespace splitting
- **Post-processing**: BERT-style processing with `[CLS]` and `[SEP]` tokens
- **Special tokens mapping**:
  - `[CLS]` → BOS token
  - `[SEP]` → EOS token

## Time Estimates

- **10M tokens**: ~5-10 minutes
- **100M tokens**: ~15-30 minutes

## Important Notes

1. **Do this once**: Train the tokenizer once and reuse it for all experiments
2. **Consistency**: Use the same tokenizer for all training runs to ensure compatibility
3. **Vocabulary size**:
   - Smaller vocab (16K): Faster training, less memory
   - Larger vocab (50K): Better coverage, more memory
   - 32K is a good default
4. **Google Drive**: Save to Google Drive so it persists across Colab sessions

## Troubleshooting

### Error: "No .train files found"
- Make sure the `--data_dir` path is correct
- Check that the directory contains `.train` files

### Error: "Permission denied" when saving
- Make sure Google Drive is mounted
- Check that the output directory exists or can be created

### Error: "Out of memory"
- Use a smaller vocabulary size (e.g., `--vocab_size 16000`)
- Process a smaller dataset first to test

## Advanced Usage

### Custom Vocabulary Size

```python
!python train_tokenizer.py \
    --data_dir /content/drive/MyDrive/babylm-edlm/data/train_10M \
    --output_path /content/drive/MyDrive/babylm-edlm/tokenizer.json \
    --vocab_size 50000 \
    --min_frequency 3
```

### Train on 100M Dataset

```python
!python train_tokenizer.py \
    --data_dir /content/drive/MyDrive/babylm-edlm/data/train_100M \
    --output_path /content/drive/MyDrive/babylm-edlm/tokenizer_100M.json \
    --vocab_size 32000
```

