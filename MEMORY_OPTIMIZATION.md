# Memory Optimization Guide for Colab A100 40GB

If you encounter CUDA out of memory errors, try these solutions:

## Quick Fix: Reduce Batch Size

The default batch size (512) may be too large. Try reducing it:

```bash
# Reduce global batch size to 128
python main.py data=babylm loader.global_batch_size=128

# Or even smaller for testing
python main.py data=babylm loader.global_batch_size=64
```

## Use Gradient Accumulation

Maintain effective batch size while reducing memory:

```bash
# Batch size 64 with gradient accumulation of 8 = effective batch size 512
python main.py data=babylm \
    loader.global_batch_size=64 \
    trainer.accumulate_grad_batches=8
```

## Reduce Sequence Length

If batch size reduction isn't enough, reduce sequence length:

```bash
# Use smaller sequence length (512 instead of 1024)
python main.py data=babylm \
    loader.global_batch_size=128 \
    model.length=512
```

## Use Smaller Model

Try the tiny model instead of small:

```bash
python main.py data=babylm model=tiny loader.global_batch_size=256
```

## Enable Memory Optimization

Set environment variable to reduce fragmentation:

```python
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

Then run training.

## Recommended Settings for A100 40GB

For the **small model** (117M params):
```bash
python main.py data=babylm \
    loader.global_batch_size=128 \
    trainer.accumulate_grad_batches=4 \
    model.length=1024
```

For the **tiny model** (if small still doesn't fit):
```bash
python main.py data=babylm \
    model=tiny \
    loader.global_batch_size=256 \
    trainer.accumulate_grad_batches=2 \
    model.length=1024
```

## Memory Usage Breakdown

- Model (small, 117M params): ~470 MB
- Per batch element (seq_len=1024): ~4-8 MB
- Batch size 512: ~2-4 GB just for activations
- With gradients and optimizer states: ~3-6x more

**Total for batch_size=512**: ~12-24 GB
**Total for batch_size=128**: ~3-6 GB

## Troubleshooting

1. **Clear GPU cache before training:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

2. **Check current memory usage:**
   ```python
   import torch
   print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
   print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
   ```

3. **Use mixed precision (already enabled):**
   - The config uses `precision: 'bf16'` which helps reduce memory

4. **Reduce validation batch size separately (IMPORTANT for validation OOM):**
   ```bash
   python main.py data=babylm \
       loader.global_batch_size=128 \
       loader.eval_global_batch_size=32  # Much smaller for validation
   ```

5. **Limit validation dataset size:**
   ```bash
   python main.py data=babylm \
       trainer.limit_val_batches=0.5  # Only validate on 50% of validation set
   ```

6. **Clear GPU cache before validation (add to training code if needed):**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

