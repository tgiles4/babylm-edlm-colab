#!/bin/bash
# Text8 MDLM backbone training (1× A100 80GB).
#SBATCH --job-name=text8_mdlm
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --gres=gpu:A100.80gb:1

#SBATCH --output=text8_mdlm-%j.out
#SBATCH --error=text8_mdlm-%j.err
#SBATCH --export=NONE
#SBATCH --constraint=amd

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=64G
#SBATCH --time=3-00:00:00

# Clone repo and set up environment (run on compute node). With --export=NONE, set SCRATCH if needed.
export SCRATCH="${SCRATCH:-/scratch/$(id -un)}"
export path=${SCRATCH}/edlm
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

# Text8 MDLM training (Austin et al. 2021; Campbell et al. 2024):
# 12-layer transformer, 12 heads, hidden 784, length 256, 1M steps, batch 512,
# cosine LR with 2000-step warmup, AdamW lr 3e-4, weight decay 0.03, channel dropout 0.05
"$ENV/bin/python" -u -m main \
  ++path=${path} \
  train_mdlm_only=true \
  data=text8 \
  model=small \
  model.length=256 \
  model.n_blocks=12 \
  model.n_heads=12 \
  model.dropout=0.05 \
  loader.global_batch_size=512 \
  trainer.max_steps=1000000 \
  trainer.val_check_interval=1.0 \
  optim.lr=0.0003 \
  optim.weight_decay=0.03 \
  lr_scheduler=cosine_decay_warmup \
  lr_scheduler.warmup_t=2000 \
  checkpointing.save_dir=${path} \
  hydra.run.dir=outputs/text8_mdlm \
  wandb.id=null \
  wandb.group=text8_mdlm \
  wandb.name=text8_mdlm
