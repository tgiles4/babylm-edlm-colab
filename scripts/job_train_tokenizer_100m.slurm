#!/bin/bash
# Train BPE tokenizer on BabyLM 100M strict corpus. CPU-only; uses same venv as job_train_babylm_hopper.
# First run: downloads BabyLM-community/BabyLM-2026-Strict from Hugging Face (network); allow enough walltime.
# Output: project_dir/tokenizer/tokenizer_100M.json; data: train_100M/*.train + *.val (see babylm_hf.py).
# Submit from repo root: sbatch scripts/job_train_tokenizer_100m.slurm
#SBATCH --partition=normal
#SBATCH --job-name=tok-100M
#SBATCH --output=tok-100M-%j.out
#SBATCH --error=tok-100M-%j.err
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=2GB

# Clone repo and set up environment (run on compute node). With --export=NONE, set SCRATCH if needed.
export SCRATCH="${SCRATCH:-/scratch/$(id -un)}"
export path=${SCRATCH}/edlm
mkdir -p ${path}
cd ${path}

if [ ! -d "babylm-edlm-colab" ]; then
  git clone git@github.com:tgiles4/babylm-edlm-colab.git
fi
cd babylm-edlm-colab

# Use all requested CPUs: tokenizers uses Rayon (Rust); limit to our allocation
export RAYON_RS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export TOKENIZERS_PARALLELISM=true

# Load modules
module load hosts/hopper
module load gnu10/10.3.0-ya
module load python/3.9.9-jh
module load cuda/12.4.0

ENV=$SCRATCH/edlm/venv
PROJECT_DIR="${SCRATCH}/babylm-edlm"

"$ENV/bin/python" -u train_tokenizer.py \
  --project_dir "$PROJECT_DIR" \
  --size 100M \
  --vocab_size 16384
