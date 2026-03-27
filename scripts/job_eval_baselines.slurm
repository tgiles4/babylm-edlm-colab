#!/bin/bash
# Evaluate external HuggingFace baselines on the BabyLM validation set.
# Computes NLL / PPL / BPD for:
#   - despoinakk/diffusion_cosine_babylm   (5 checkpoints, diffusion ELBO)
#   - BabyLM-community/babylm-baseline-100m-gpt2  (causal NLL)
#   - BabyLM-community/babylm-baseline-100m-gpt-bert-masked-focus  (3 methods)
#   - BabyLM-community/babylm-baseline-100m-gpt-bert-causal-focus  (3 methods)
#
# Submit:  sbatch scripts/job_eval_baselines.slurm
#SBATCH --job-name=eval_baselines
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --gres=gpu:A100.80gb:1

#SBATCH --output=eval_baselines-%j.out
#SBATCH --error=eval_baselines-%j.err
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

# Dedicated eval venv with modern HuggingFace (no flash-attn needed)
EVAL_ENV=${SCRATCH}/edlm/eval_venv
if [ ! -f "${EVAL_ENV}/bin/activate" ]; then
  echo "Creating eval venv at ${EVAL_ENV} ..."
  python3 -m venv "${EVAL_ENV}"
  "${EVAL_ENV}/bin/pip" install --upgrade pip
  "${EVAL_ENV}/bin/pip" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  "${EVAL_ENV}/bin/pip" install transformers tokenizers huggingface_hub torchmetrics
fi
PY="${EVAL_ENV}/bin/python"

# HuggingFace cache inside scratch so downloads persist between jobs
export HF_HOME=${SCRATCH}/hf_cache
mkdir -p ${HF_HOME}

# Load HF token if available (for gated repos)
if [ -f "${HOME}/.hf_token" ]; then
  export HF_TOKEN=$(cat "${HOME}/.hf_token")
fi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR=${SCRATCH}/babylm-edlm/data/train_100M
RESULTS=${SCRATCH}/edlm/eval_results
mkdir -p ${RESULTS}

BATCH_SIZE=1024
SEQ_LENGTH=512

# ---------------------------------------------------------------------------
# Helper: run eval_baselines.py and print timing
# ---------------------------------------------------------------------------
run_eval() {
  echo ""
  echo "================================================================"
  echo "  $@"
  echo "================================================================"
  $PY -u eval_baselines.py "$@" 2>&1
  echo "  -> exit code: $?"
}

# ---------------------------------------------------------------------------
# 1. despoinakk/diffusion_cosine_babylm  (5 checkpoints)
# ---------------------------------------------------------------------------
echo "===== Diffusion (despoinakk) ====="
DIFF_REPO="despoinakk/diffusion_cosine_babylm"

run_eval \
  --model_type diffusion \
  --model_name_or_path ${DIFF_REPO} \
  --checkpoint_subfolder chck_1000M \
  --data_dir ${DATA_DIR} \
  --batch_size ${BATCH_SIZE} \
  --seq_length ${SEQ_LENGTH} \
  --output_json ${RESULTS}/diffusion_chck_1000M.json

# ---------------------------------------------------------------------------
# 2. BabyLM-community/babylm-baseline-100m-gpt2
# ---------------------------------------------------------------------------
echo ""
echo "===== GPT-2 Baseline ====="
run_eval \
  --model_type ar \
  --model_name_or_path BabyLM-community/babylm-baseline-100m-gpt2 \
  --data_dir ${DATA_DIR} \
  --batch_size ${BATCH_SIZE} \
  --seq_length ${SEQ_LENGTH} \
  --output_json ${RESULTS}/gpt2.json

# ---------------------------------------------------------------------------
# 3. GPT-BERT masked-focus  (causal + masked_ce + pll)
# ---------------------------------------------------------------------------
echo ""
echo "===== GPT-BERT Masked Focus ====="
run_eval \
  --model_type gpt_bert \
  --model_name_or_path BabyLM-community/babylm-baseline-100m-gpt-bert-masked-focus \
  --data_dir ${DATA_DIR} \
  --batch_size ${BATCH_SIZE} \
  --seq_length ${SEQ_LENGTH} \
  --methods causal,masked_ce,pll \
  --pll_spacing 64 \
  --output_json ${RESULTS}/gpt_bert_masked_focus.json

# ---------------------------------------------------------------------------
# 4. GPT-BERT causal-focus  (causal + masked_ce + pll)
# ---------------------------------------------------------------------------
echo ""
echo "===== GPT-BERT Causal Focus ====="
run_eval \
  --model_type gpt_bert \
  --model_name_or_path BabyLM-community/babylm-baseline-100m-gpt-bert-causal-focus \
  --data_dir ${DATA_DIR} \
  --batch_size ${BATCH_SIZE} \
  --seq_length ${SEQ_LENGTH} \
  --methods causal,masked_ce,pll \
  --pll_spacing 64 \
  --output_json ${RESULTS}/gpt_bert_causal_focus.json

# ---------------------------------------------------------------------------
# 5. Aggregate results into a summary table
# ---------------------------------------------------------------------------
echo ""
echo "===== Aggregating Results ====="
$PY -u -c "
import json, glob, os, csv

results_dir = '${RESULTS}'
rows = []

for jf in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
    with open(jf) as f:
        data = json.load(f)
    model = data.get('model_name_or_path', '')
    sub = data.get('checkpoint_subfolder', '') or ''
    res = data.get('results', {})

    # GPT-BERT has nested results per method
    if isinstance(res, dict) and any(isinstance(v, dict) for v in res.values()):
        for method_key, method_res in res.items():
            if isinstance(method_res, dict):
                rows.append({
                    'model': model,
                    'checkpoint': sub,
                    'method': method_res.get('method', method_key),
                    'nll': method_res.get('nll', ''),
                    'ppl': method_res.get('ppl', ''),
                    'bpd': method_res.get('bpd', ''),
                })
    else:
        rows.append({
            'model': model,
            'checkpoint': sub,
            'method': res.get('method', ''),
            'nll': res.get('nll', ''),
            'ppl': res.get('ppl', ''),
            'bpd': res.get('bpd', ''),
        })

summary = os.path.join(results_dir, 'summary.csv')
with open(summary, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['model', 'checkpoint', 'method', 'nll', 'ppl', 'bpd'])
    w.writeheader()
    w.writerows(rows)

print(f'Summary written to {summary}  ({len(rows)} rows)')
print()
print(f'{\"model\":>55s}  {\"ckpt\":>8s}  {\"method\":>24s}  {\"NLL\":>8s}  {\"PPL\":>10s}  {\"BPD\":>8s}')
print('-' * 120)
for r in rows:
    print(f'{r[\"model\"]:>55s}  {r[\"checkpoint\"]:>8s}  {r[\"method\"]:>24s}  {r[\"nll\"]:>8.4f}  {r[\"ppl\"]:>10.2f}  {r[\"bpd\"]:>8.4f}' if isinstance(r['nll'], float) else f'{r[\"model\"]:>55s}  {r[\"checkpoint\"]:>8s}  {r[\"method\"]:>24s}  {str(r[\"nll\"]):>8s}  {str(r[\"ppl\"]):>10s}  {str(r[\"bpd\"]):>8s}')
"

echo ""
echo "===== All evaluations complete ====="
echo "Results directory: ${RESULTS}"
