#!/bin/bash
# Resume BabyLM MDLM training from the best checkpoint (2× A100 80GB DDP).
# Finds the best-*.ckpt with the lowest val/nll and resumes from it.
# Submit from repo root: sbatch scripts/job_resume_babylm_hopper.slurm
#SBATCH --job-name=babylm_mdlm_resume
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --gres=gpu:A100.80gb:2

#SBATCH --output=babylm_mdlm_resume-%j.out
#SBATCH --error=babylm_mdlm_resume-%j.err
#SBATCH --export=NONE
#SBATCH --constraint=amd

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2

#SBATCH --mem=64G
#SBATCH --time=3-00:00:00

export SCRATCH="${SCRATCH:-/scratch/$(id -un)}"
export path=${SCRATCH}/edlm

CKPT_DIR="/scratch/tgiles/babylm-edlm/checkpoints/babylm/100M/False/2026.03.25/014557"
CKPT_SUBDIR="${CKPT_DIR}/checkpoints"

# Find the best checkpoint (lowest val/nll value in filename).
BEST_CKPT=$(ls "${CKPT_SUBDIR}"/best-*.ckpt 2>/dev/null | sort -t'-' -k2 -n | head -1)
if [ -z "${BEST_CKPT}" ]; then
  echo "ERROR: No best-*.ckpt found in ${CKPT_SUBDIR}"
  echo "Available checkpoints:"
  ls -la "${CKPT_SUBDIR}"/*.ckpt 2>/dev/null
  exit 1
fi
echo "Resuming from best checkpoint: ${BEST_CKPT}"

mkdir -p ${path}
cd ${path}

if [ ! -d "babylm-edlm-colab" ]; then
  git clone git@github.com:tgiles4/babylm-edlm-colab.git
fi
cd babylm-edlm-colab

module load hosts/hopper
module load gnu10/10.3.0-ya
module load python/3.9.9-jh
module load cuda/12.4.0

ENV=$SCRATCH/edlm/venv

if [ -f "${HOME}/.wandb_api_key" ]; then
  export WANDB_API_KEY=$(cat "${HOME}/.wandb_api_key")
fi

srun --ntasks=2 --ntasks-per-node=2 --mpi=pmi2 --export=ALL "$ENV/bin/python" -u -m main \
  ++path=${path} \
  use_energy=False \
  data=babylm \
  model=small \
  model.length=512 \
  dataset_size=100M \
  loader.global_batch_size=448 \
  loader.batch_size=112 \
  loader.eval_batch_size=128 \
  loader.num_workers=0 \
  trainer.max_epochs=10 \
  trainer.val_check_interval=1.0 \
  trainer.gradient_clip_val=1.0 \
  optim.lr=0.007 \
  optim.weight_decay=0.1 \
  lr_scheduler=cosine_decay_warmup \
  checkpointing.resume_from_ckpt=true \
  checkpointing.resume_ckpt_path="${BEST_CKPT}" \
  checkpointing.save_dir="${CKPT_DIR}" \
  wandb.id=l37daexr \
  wandb.group=babylm_mdlm \
  wandb.name=babylm_100M_resumed
