#!/bin/bash
# Train BPE tokenizer on BabyLM 10M corpus. CPU-only; uses same venv as job_train_babylm_hopper.
# 10M: no Hub auto-download — populate project_dir/data/train_10M/*.train first (strict-small or subsample).
# Output: project_dir/tokenizer/tokenizer_10M.json (default project_dir=$SCRATCH/babylm-edlm).
# Submit from repo root: sbatch scripts/job_train_tokenizer_10m.slurm
#SBATCH --partition=normal
#SBATCH --job-name=tok-10M
#SBATCH --output=tok-10M-%j.out
#SBATCH --error=tok-10M-%j.err
#SBATCH --time=0-02:00:00
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

module load hosts/hopper
module load gnu10/10.3.0-ya
module load python/3.9.9-jh
module load cuda/12.4.0

ENV=$SCRATCH/edlm/venv
PROJECT_DIR="${SCRATCH}/babylm-edlm"

"$ENV/bin/python" -u train_tokenizer.py \
  --project_dir "$PROJECT_DIR" \
  --size 10M
