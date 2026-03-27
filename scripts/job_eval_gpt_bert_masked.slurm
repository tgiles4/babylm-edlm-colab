#!/bin/bash
# Evaluate GPT-BERT masked-focus baseline on the BabyLM validation set.
# Computes NLL / PPL / BPD via causal, masked_ce, and pll methods.
#
# Submit:  sbatch scripts/job_eval_gpt_bert_masked.slurm
#SBATCH --job-name=eval_gptbert_masked
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --gres=gpu:A100.80gb:1

#SBATCH --output=eval_gptbert_masked-%j.out
#SBATCH --error=eval_gptbert_masked-%j.err
#SBATCH --export=NONE
#SBATCH --constraint=amd

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export SCRATCH="${SCRATCH:-/scratch/$(id -un)}"
export EDLM=${SCRATCH}/edlm
mkdir -p ${EDLM}
cd ${EDLM}

if [ ! -d "babylm-edlm-colab" ]; then
  git clone git@github.com:tgiles4/babylm-edlm-colab.git
fi
cd babylm-edlm-colab
git pull --ff-only 2>/dev/null || true

module load hosts/hopper
module load gnu10/10.3.0-ya
module load python/3.9.9-jh
module load cuda/12.4.0

EVAL_ENV=${SCRATCH}/edlm/eval_venv
if [ ! -f "${EVAL_ENV}/bin/activate" ]; then
  echo "Creating eval venv at ${EVAL_ENV} ..."
  python3 -m venv "${EVAL_ENV}"
  "${EVAL_ENV}/bin/pip" install --upgrade pip
  "${EVAL_ENV}/bin/pip" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  "${EVAL_ENV}/bin/pip" install transformers tokenizers huggingface_hub torchmetrics
fi
PY="${EVAL_ENV}/bin/python"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=${SCRATCH}/hf_cache
mkdir -p ${HF_HOME}

if [ -f "${HOME}/.hf_token" ]; then
  export HF_TOKEN=$(cat "${HOME}/.hf_token")
fi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR=${SCRATCH}/babylm-edlm/data/train_100M
RESULTS=${SCRATCH}/edlm/eval_results
mkdir -p ${RESULTS}

BATCH_SIZE=512
SEQ_LENGTH=512

# ---------------------------------------------------------------------------
# GPT-BERT masked-focus  (causal + masked_ce + pll)
# ---------------------------------------------------------------------------
echo "===== GPT-BERT Masked Focus ====="
$PY -u eval_baselines.py \
  --model_type gpt_bert \
  --model_name_or_path BabyLM-community/babylm-baseline-100m-gpt-bert-masked-focus \
  --data_dir ${DATA_DIR} \
  --batch_size ${BATCH_SIZE} \
  --seq_length ${SEQ_LENGTH} \
  --methods causal,masked_ce,pll \
  --pll_spacing 64 \
  --output_json ${RESULTS}/gpt_bert_masked_focus.json 2>&1

echo "  -> exit code: $?"
echo ""
echo "===== GPT-BERT Masked Focus complete ====="
