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

# Clone repo and set up environment (run on compute node).
mkdir -p ${path}
cd ${path}

if [ ! -d "babylm-edlm-colab" ]; then
  git clone git@github.com:tgiles4/babylm-edlm-colab.git
fi
cd babylm-edlm-colab

# Load modules
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

# 2-GPU DDP via srun. Uses --mpi=pmi2 to work around cluster-wide PMIx hang (SLURM 24.11.5).
# loader.num_workers=0 avoids DataLoader BrokenPipe with multi-GPU DDP.
# --export=ALL passes WANDB_API_KEY to all tasks.
srun --ntasks=2 --ntasks-per-node=2 --mpi=pmi2 --export=ALL "$ENV/bin/python" -u -m main \
  ++path=${path} \
  use_energy=True \
  data=babylm \
  model=small \
  model.length=256 \
  dataset_size=100M \
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
  wandb.name=babylm_100M_energy
