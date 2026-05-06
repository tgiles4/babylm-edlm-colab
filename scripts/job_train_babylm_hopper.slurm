#!/bin/bash
# BabyLM MDLM only (no energy): DIT diffusion on Hopper.
# Requests 4× A100 80GB across 2 nodes (2 GPUs/node).
# BabyLM rules: 10 epoch total budget.
# Paths use $SCRATCH/babylm-edlm (see configs and HOPPER_SETUP.md).
# Checkpoints: ${BABYLM_CHECKPOINT_ROOT}/checkpoints/${BABYLM_RUN_NAME}/ (Lightning ModelCheckpoint dirpath).
# Energy follow-up: same BABYLM_RUN_NAME -> checkpoints/${BABYLM_RUN_NAME}_energy/ (job_train_babylm_energy_hopper.slurm).
#
# Partition notes:
#   - gpuq: non-pre-emptible
#   - contrib-gpuq: same nodes, may be pre-empted (often faster queue time)
#
# Submit examples:
#   sbatch scripts/job_train_babylm_hopper.slurm
#   sbatch --export=NONE,BABYLM_RUN_NAME=my_run scripts/job_train_babylm_hopper.slurm
#
#SBATCH --job-name=babylm_mdlm
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --gres=gpu:A100.80gb:2
#SBATCH --output=babylm_mdlm-%j.out
#SBATCH --error=babylm_mdlm-%j.err
#SBATCH --export=NONE
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --constraint=amd
#SBATCH --mem=64G
#SBATCH --time=3-0:00:00

# Clone repo and set up environment (run on compute node). With --export=NONE, set SCRATCH if needed.
export SCRATCH="${SCRATCH:-/scratch/$(id -un)}"
export path=${SCRATCH}/edlm
# Parent of checkpoints/ (matches project_dir default in configs when using $SCRATCH/babylm-edlm).
export BABYLM_CHECKPOINT_ROOT="${BABYLM_CHECKPOINT_ROOT:-${SCRATCH}/babylm-edlm}"
# --- Edit this name per experiment (no timestamp). Energy training uses the same name + _energy. ---
BABYLM_RUN_NAME="${BABYLM_RUN_NAME:-babylm_strict}"
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

# 4-GPU multi-node DDP via srun. Uses --mpi=pmi2 to work around cluster-wide PMIx hang (SLURM 24.11.5).
# global_batch_size=512 with batch_size=128 on 2×2 GPUs -> accumulate_grad_batches=1 (see configs/config.yaml).
# loader.num_workers=0 avoids DataLoader BrokenPipe with multi-GPU DDP.
srun --ntasks=4 --ntasks-per-node=2 --mpi=pmi2 --export=ALL "$ENV/bin/python" -u -m main \
  ++path=${path} \
  trainer.num_nodes=2 \
  trainer.devices=2 \
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
  trainer.val_check_interval=0.25 \
  trainer.gradient_clip_val=2.0 \
  optim.lr=0.007 \
  optim.weight_decay=0.1 \
  lr_scheduler=cosine_decay_warmup \
  checkpointing.save_dir=${CHECKPOINT_SAVE_DIR} \
  checkpointing.resume_from_ckpt=true \
  checkpointing.resume_ckpt_path=${CHECKPOINT_SAVE_DIR}/last.ckpt \
  callbacks.checkpoint_monitor.save_last=true \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=200 \
  wandb.id=null \
  wandb.group=babylm_mdlm \
  wandb.name=${BABYLM_RUN_NAME} \
  'wandb.notes=Base MDLM for BabyLM 100M 10 epochs'
