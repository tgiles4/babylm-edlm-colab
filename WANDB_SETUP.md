# Weights & Biases (wandb) Setup Guide

## Quick Setup for Colab

### Step 1: Login to wandb

In a Colab cell, run:
```python
import wandb
wandb.login()
```

Enter your API key when prompted.

### Step 2: Verify wandb is working

```python
import wandb
wandb.init(project="test", mode="disabled")
print("✓ wandb is configured correctly")
```

### Step 3: Check if metrics are being logged

During training, you should see wandb output like:
```
wandb: Currently logged in as: [your-username]
wandb: Tracking run with wandb version [version]
wandb: Run data is saved locally in: [path]
wandb: Syncing run [run-name] to [project]
```

## Troubleshooting

### Issue: Metrics not showing up in wandb dashboard

**Possible causes:**

1. **wandb not initialized properly**
   - Check that `wandb.login()` was successful
   - Verify you see "wandb: Currently logged in as: ..." in output

2. **Run name/id issues**
   - The config now auto-generates names from your settings
   - Check the wandb output for the run name

3. **Offline mode**
   - If `wandb.offline: true`, metrics are saved locally but not synced
   - Set `wandb.offline: false` or remove the setting

4. **Check local wandb directory**
   - wandb logs are saved to: `/content/drive/MyDrive/babylm-edlm/wandb`
   - You can sync later with: `wandb sync /path/to/wandb/run`

### Enable wandb explicitly

If metrics still don't show, you can force wandb initialization:

```python
# Before running training, in a Colab cell:
import wandb
import os

# Set wandb environment variables
os.environ['WANDB_PROJECT'] = 'text-diffusion'
os.environ['WANDB_ENTITY'] = 'your-entity'  # Optional: your wandb team/username

# Verify
wandb.init(project="test", mode="disabled")
print("✓ wandb ready")
```

### Check wandb status during training

Look for these log messages in your training output:
- `wandb: Tracking run with wandb version ...`
- `wandb: Run data is saved locally in: ...`
- `wandb: Syncing run ... to [project]`

### View metrics

1. **Online**: Go to https://wandb.ai and check your project
2. **Local**: Check `/content/drive/MyDrive/babylm-edlm/wandb/` directory
3. **Sync later**: If offline, run `wandb sync /path/to/run` after training

## Configuration Options

You can override wandb settings via command line:

```bash
# Change project name
python main.py wandb.project=babylm-training

# Set entity/team
python main.py wandb.entity=your-team

# Enable offline mode (sync later)
python main.py wandb.offline=true

# Set custom run name
python main.py wandb.name=my-experiment-1
```

## Metrics Being Logged

The codebase logs:
- Training metrics: `train/nll`, `train/bpd`, `train/ppl`
- Validation metrics: `val/nll`, `val/bpd`, `val/ppl`
- Learning rate: `lr-AdamW`
- Model checkpoints (if configured)

Check your wandb dashboard to see these metrics updating in real-time.





