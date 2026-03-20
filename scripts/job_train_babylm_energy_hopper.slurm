#!/bin/bash
# BabyLM EBM (with energy): finetune from a chosen diffusion checkpoint on Hopper (2× A100 80GB).
# MDLM is trained for 10 epochs (full BabyLM budget); you choose which checkpoint to load (e.g. last.ckpt,
# or an earlier checkpoint from that run). Requires at least 2 GPUs.
# Paths use $SCRATCH/babylm-edlm (see configs and HOPPER_SETUP.md).
# Submit: sbatch --export=NONE,BABYLM_CKPT=/path/to/checkpoint.ckpt scripts/job_train_babylm_energy_hopper.slurm
#SBATCH --job-name=babylm_energy
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --gres=gpu:A100.80gb:2

#SBATCH --output=babylm_energy-%j.out
#SBATCH --error=babylm_energy-%j.err
#SBATCH --export=NONE
#SBATCH --constraint=amd

#SBATCH --nodes=1
# Must match gres GPU count for DDP.
#SBATCH --ntasks-per-node=2

#SBATCH --mem=64G
#SBATCH --time=3-00:00:00

export SCRATCH="${SCRATCH:-/scratch/$(id -un)}"
export path=${SCRATCH}/edlm
# Which diffusion checkpoint to load as EBM backbone. MDLM is trained for 10 epochs; choose any checkpoint
# from that run (e.g. last.ckpt for final, or a step checkpoint for an earlier point).
export BABYLM_CKPT="${BABYLM_CKPT:-}"
if [ -z "${BABYLM_CKPT}" ]; then
  echo "ERROR: BABYLM_CKPT must be set to a diffusion checkpoint from your MDLM run."
  echo "Example: sbatch --export=NONE,BABYLM_CKPT=/path/to/checkpoints/.../last.ckpt scripts/job_train_babylm_energy_hopper.slurm"
  exit 1
fi

# Clone repo and set working directory (same pattern as scripts/job_text8_mdlm_4gpu.slurm).
# Optional: sbatch --export=NONE,BABYLM_CKPT=...,EDLM_BRANCH=my-feature ...
mkdir -p ${path}
cd ${path}

EDLM_REPO="${EDLM_REPO:-https://github.com/tgiles4/babylm-edlm-colab.git}"
EDLM_DIR="${EDLM_DIR:-babylm-edlm-colab}"

if [ ! -d "${EDLM_DIR}" ]; then
  if [ -n "${EDLM_BRANCH:-}" ]; then
    git clone -b "${EDLM_BRANCH}" "${EDLM_REPO}" "${EDLM_DIR}"
  else
    git clone "${EDLM_REPO}" "${EDLM_DIR}"
  fi
fi
cd "${EDLM_DIR}"

if [ -n "${EDLM_BRANCH:-}" ]; then
  git fetch origin "${EDLM_BRANCH}" 2>/dev/null || git fetch origin
  git checkout "${EDLM_BRANCH}"
  git pull --ff-only origin "${EDLM_BRANCH}" 2>/dev/null || true
fi

# Load modules (match job_text8_nce_4gpu.slurm)
module load hosts/hopper
module load gnu10/10.3.0-ya
module load python/3.9.9-jh
module load cuda/12.4.0

ENV=$SCRATCH/edlm/venv

# W&B: load API key from a file (avoids putting secrets in the script or sbatch env).
if [ -f "${HOME}/.wandb_api_key" ]; then
  export WANDB_API_KEY=$(cat "${HOME}/.wandb_api_key")
fi

# BabyLM EBM (use_energy=True): load diffusion backbone from checkpoint, train with energy.
# loader.num_workers=0 avoids DataLoader BrokenPipe with multi-GPU.
# srun launches one process per task so DDP uses both GPUs; --export=ALL passes WANDB_API_KEY to all tasks.
srun --ntasks=2 --ntasks-per-node=2 --export=ALL "$ENV/bin/python" -u -m main \
  ++path=${path} \
  use_energy=True \
  data=babylm \
  model=small \
  model.length=256 \
  dataset_size=10M \
  noise=cosine \
  eval.checkpoint_path=${BABYLM_CKPT} \
  ebm_backbone=dit \
  loader.global_batch_size=512 \
  loader.num_workers=0 \
  trainer.max_steps=10000 \
  trainer.val_check_interval=1.0 \
  optim.lr=0.0003 \
  optim.weight_decay=0.03 \
  lr_scheduler=cosine_decay_warmup \
  lr_scheduler.warmup_t=2000 \
  checkpointing.resume_from_ckpt=false \
  wandb.id=null \
  wandb.group=babylm_energy \
  wandb.name=babylm_10M_energy
