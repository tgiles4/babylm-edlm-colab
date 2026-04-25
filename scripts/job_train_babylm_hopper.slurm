#!/bin/bash
# BabyLM MDLM only (no energy): DIT diffusion training on Hopper (2× A100 80GB).
# BabyLM rules: 10 epoch total budget.
# Paths use $SCRATCH/babylm-edlm (see configs and HOPPER_SETUP.md).
# Submit from repo root: sbatch scripts/job_train_babylm_hopper.slurm
# Checkpoints: edit BABYLM_RUN_NAME (or sbatch --export=NONE,BABYLM_RUN_NAME=...).
# Diffusion: ${BABYLM_CHECKPOINT_ROOT}/checkpoints/${BABYLM_RUN_NAME}/ (Lightning ModelCheckpoint dirpath).
# Energy job: same BABYLM_RUN_NAME -> saves under checkpoints/${BABYLM_RUN_NAME}_energy/...
#SBATCH --job-name=babylm_mdlm
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --gres=gpu:A100.80gb:2

#SBATCH --output=babylm_mdlm-%j.out
#SBATCH --error=babylm_mdlm-%j.err
#SBATCH --export=NONE
#SBATCH --constraint=amd

#SBATCH --nodes=1
# Must match gres GPU count for DDP.
#SBATCH --ntasks-per-node=2

#SBATCH --mem=64G
#SBATCH --time=3-00:00:00

# Clone repo and set up environment (run on compute node). With --export=NONE, set SCRATCH if needed.
export SCRATCH="${SCRATCH:-/scratch/$(id -un)}"
export path=${SCRATCH}/edlm
# Parent of checkpoints/ (matches project_dir default in configs when using $SCRATCH/babylm-edlm).
export BABYLM_CHECKPOINT_ROOT="${BABYLM_CHECKPOINT_ROOT:-${SCRATCH}/babylm-edlm}"
# --- Edit this name per experiment (no timestamp). Energy training uses the same name + _energy. ---
BABYLM_RUN_NAME="${BABYLM_RUN_NAME:-babylm_100M}"
CHECKPOINT_SAVE_DIR="${BABYLM_CHECKPOINT_ROOT}/checkpoints/${BABYLM_RUN_NAME}"
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

echo "checkpointing.save_dir -> ${CHECKPOINT_SAVE_DIR}"

# 2-GPU DDP via srun. Uses --mpi=pmi2 to work around cluster-wide PMIx hang (SLURM 24.11.5).
# global_batch_size=512 with batch_size=128 -> accumulate_grad_batches=2 (see configs/config.yaml).
# loader.num_workers=0 avoids DataLoader BrokenPipe with multi-GPU DDP.
srun --ntasks=2 --ntasks-per-node=2 --mpi=pmi2 --export=ALL "$ENV/bin/python" -u -m main \
  ++path=${path} \
  use_energy=False \
  data=babylm \
  model=small \
  model.length=512 \
  dataset_size=100M \
  loader.global_batch_size=512 \
  loader.batch_size=128 \
  loader.eval_batch_size=128 \
  loader.num_workers=0 \
  trainer.max_epochs=10 \
  trainer.val_check_interval=1.0 \
  trainer.gradient_clip_val=2.0 \
  optim.lr=0.007 \
  optim.weight_decay=0.1 \
  lr_scheduler=cosine_decay_warmup \
  checkpointing.save_dir=${CHECKPOINT_SAVE_DIR} \
  checkpointing.resume_from_ckpt=true \
  checkpointing.resume_ckpt_path=${CHECKPOINT_SAVE_DIR}/last.ckpt \
  callbacks.checkpoint_monitor.save_last=true \
  wandb.id=null \
  wandb.group=babylm_mdlm \
  wandb.name=${BABYLM_RUN_NAME} \
  wandb.notes=Base MDLM for BabyLM 100M 10 epochs
