#!/bin/bash
# BabyLM MDLM only (no energy): DIT diffusion training on Hopper (2× A100 80GB).
# BabyLM rules: 10 epoch total budget. This script trains the backbone for 7 epochs,
# leaving 3 epochs for the energy NCE model (job_train_babylm_energy_hopper.slurm).
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
#SBATCH --time=0-12:00:00

# Clone repo and set up environment (run on compute node). With --export=NONE, set SCRATCH if needed.
export SCRATCH="${SCRATCH:-/scratch/$(id -un)}"
export path=${SCRATCH}/edlm
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

# Single GPU while srun multi-task is broken cluster-wide.
# batch_size=256 fits on 1x A100 80GB (~70-80% VRAM); accumulate_grad_batches
# auto-computes to 2 for effective global_batch_size=512.
"$ENV/bin/python" -u -m main \
  ++path=${path} \
  use_energy=False \
  data=babylm \
  model=small \
  model.length=512 \
  dataset_size=100M \
  loader.global_batch_size=512 \
  loader.batch_size=256 \
  loader.eval_batch_size=256 \
  loader.num_workers=0 \
  trainer.max_epochs=10 \
  trainer.val_check_interval=1.0 \
  trainer.gradient_clip_val=1.0 \
  optim.lr=0.007 \
  optim.weight_decay=0.1 \
  lr_scheduler=cosine_decay_warmup \
  wandb.id=null \
  wandb.group=babylm_mdlm \
  wandb.name=babylm_100M
