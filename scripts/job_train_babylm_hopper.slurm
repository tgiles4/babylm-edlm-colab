#!/bin/bash
# BabyLM MDLM only (no energy): DIT diffusion training on Hopper (1× A100 80GB).
# BabyLM rules: train for 10 epochs only. This script uses the full 10 epochs; the EBM script
# then chooses which checkpoint from this run to load as backbone (early, mid, or final).
# Paths use $SCRATCH/babylm-edlm (see configs and HOPPER_SETUP.md).
# Submit from repo root: sbatch scripts/job_train_babylm_hopper.slurm
# For EBM (with energy) use job_train_babylm_energy_hopper.slurm and set BABYLM_CKPT to desired checkpoint.
#SBATCH --job-name=babylm_mdlm
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --gres=gpu:A100.80gb:1

#SBATCH --output=babylm_mdlm-%j.out
#SBATCH --error=babylm_mdlm-%j.err
#SBATCH --export=NONE
#SBATCH --constraint=amd

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=64G
#SBATCH --time=3-00:00:00

# With --export=NONE, set SCRATCH so config project_dir ($SCRATCH/babylm-edlm) resolves on the compute node.
export SCRATCH="${SCRATCH:-/scratch/$(id -un)}"
export path=${SCRATCH}/edlm

# Run from repo root (directory containing this script's parent)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Load modules (match job_text8_mdlm_4gpu.slurm)
module load hosts/hopper
module load gnu10/10.3.0-ya
module load python/3.9.9-jh
module load cuda/12.4.0

ENV=$SCRATCH/edlm/venv

# W&B: load API key from a file (avoids putting secrets in the script or sbatch env).
# Create once: echo "YOUR_WANDB_API_KEY" > ~/.wandb_api_key && chmod 600 ~/.wandb_api_key
if [ -f "${HOME}/.wandb_api_key" ]; then
  export WANDB_API_KEY=$(cat "${HOME}/.wandb_api_key")
fi

# BabyLM DIT (MDLM only, no energy): data=babylm, 10M corpus, cosine noise. Full 10 epochs (BabyLM budget).
# Paths (data_dir, tokenizer, cache, save_dir, hydra.run.dir) use config defaults from project_dir=$SCRATCH/babylm-edlm.
"$ENV/bin/python" -u -m main \
  ++path=${path} \
  use_energy=False \
  data=babylm \
  model=small \
  model.length=256 \
  dataset_size=10M \
  noise=cosine \
  loader.global_batch_size=512 \
  trainer.max_epochs=10 \
  trainer.val_check_interval=1.0 \
  optim.lr=0.0003 \
  optim.weight_decay=0.03 \
  lr_scheduler=cosine_decay_warmup \
  lr_scheduler.warmup_t=2000 \
  wandb.id=null \
  wandb.group=babylm_mdlm \
  wandb.name=babylm_10M
