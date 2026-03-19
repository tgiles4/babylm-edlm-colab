#!/bin/bash
# Text8 NCE EBM finetuning from MDLM checkpoint (2× A100 80GB; ntasks must match for DDP).
# Run after job_text8_mdlm_4gpu.slurm. Use loader.num_workers=0 to avoid DataLoader BrokenPipe with multi-GPU.
# Partition: gpuq = non-pre-emptible; contrib-gpuq = same nodes, can be pre-empted (use if gpuq queue is long).
# sgpu: (A)llocated, (I)dle — need I:2 for 2 free GPUs.
#SBATCH --job-name=text8_nce
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --gres=gpu:A100.80gb:2

#SBATCH --output=text8_nce-%j.out
#SBATCH --error=text8_nce-%j.err
#SBATCH --export=NONE

#SBATCH --nodes=1
# Must match gres GPU count so Lightning SLURM check passes (devices=2).
#SBATCH --ntasks-per-node=2
#SBATCH --constraint=amd

#SBATCH --mem=64G
#SBATCH --time=0-05:00:00

# Set path and MDLM checkpoint (run after job_text8_mdlm_4gpu.slurm). With --export=NONE, set SCRATCH if needed.
export SCRATCH="${SCRATCH:-/scratch/$(id -un)}"
export path=${SCRATCH}/edlm
# Path to MDLM checkpoint from script 1 (default: last.ckpt from same run).
# For a specific step (e.g. 50k): MDLM_CKPT=${path}/checkpoints/checkpoint_step=50000.ckpt
# Override when submitting: sbatch --export=NONE,MDLM_CKPT=/path/to/checkpoint.ckpt job_text8_nce_4gpu.slurm
export MDLM_CKPT="${MDLM_CKPT:-${path}/checkpoints/last.ckpt}"

mkdir -p ${path}
cd ${path}

if [ ! -d "Energy-Diffusion-LLM" ]; then
  git clone https://github.com/tgiles4/Energy-Diffusion-LLM.git
fi
cd Energy-Diffusion-LLM

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

# NCE EBM finetuning from MDLM: pooling + scalar energy head, 10k steps,
# cosine LR with 2000-step warmup, same optimizer settings.
# srun launches one process per task so DDP uses both GPUs (ntasks-per-node=2).
# --export=ALL passes WANDB_API_KEY (and other vars set above) to both tasks so rank 1 can log.
srun --ntasks=2 --ntasks-per-node=2 --export=ALL "$ENV/bin/python" -u -m main \
  ++path=${path} \
  train_mdlm_only=false \
  data=text8 \
  model=small \
  model.length=256 \
  model.hidden_size=768 \
  model.n_blocks=12 \
  model.n_heads=12 \
  model.dropout=0.05 \
  eval.checkpoint_path=${MDLM_CKPT} \
  ebm_backbone=dit \
  loader.global_batch_size=512 \
  loader.num_workers=0 \
  trainer.max_steps=10000 \
  trainer.val_check_interval=1.0 \
  optim.lr=0.0003 \
  optim.weight_decay=0.03 \
  lr_scheduler=cosine_decay_warmup \
  lr_scheduler.warmup_t=2000 \
  checkpointing.save_dir=${path} \
  checkpointing.resume_from_ckpt=false \
  hydra.run.dir=outputs/text8_nce \
  wandb.id=null \
  wandb.group=text8_nce \
  wandb.name=text8_nce
