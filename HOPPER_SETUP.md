# Hopper cluster setup (BabyLM-EDLM)

Setup for running BabyLM-EDLM on GMU's Hopper supercomputer. Conventions match the reference scripts `scripts/job_text8_mdlm_4gpu.slurm` and `scripts/job_text8_nce_4gpu.slurm`.

## Module load

Load these modules before building the environment or running jobs:

```bash
module load hosts/hopper
module load gnu10/10.3.0-ya
module load python/3.9.9-jh
module load cuda/12.4.0
```

## Conda / venv

Use a virtual environment at a fixed path so SLURM scripts can call it reliably. Example: venv at `$SCRATCH/edlm/venv` with Python 3.9 and PyTorch 2.6 (CUDA 12.4).

**One-time setup (on a login or compute node after loading modules):**

```bash
export SCRATCH="${SCRATCH:-/scratch/$(id -un)}"
export path=${SCRATCH}/edlm
mkdir -p ${path}
python -m venv ${path}/venv
source ${path}/venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# Install flash-attn, causal-conv1d, mamba-ssm if required by your run
```

Job scripts can then use:

```bash
ENV=$SCRATCH/edlm/venv
"$ENV/bin/python" -u -m main ...
```

With `#SBATCH --export=NONE`, set `SCRATCH` in the script (e.g. `export SCRATCH="${SCRATCH:-/scratch/$(id -un)}"`) so the venv path resolves on the compute node.

## Where data and checkpoints live

- **Working directory:** `$SCRATCH`. When using `#SBATCH --export=NONE`, set `SCRATCH` in the job script (e.g. `export SCRATCH="${SCRATCH:-/scratch/$(id -un)}"`).
- **Project base (config default):** `project_dir` defaults to `$SCRATCH/babylm-edlm` when `SCRATCH` is set. All paths below are under that base unless overridden.

Create this layout under `$SCRATCH/babylm-edlm` (or your overridden `project_dir`):

```
$SCRATCH/babylm-edlm/
├── data/
│   ├── train_10M/     # BabyLM .train files for 10M-token corpus
│   └── train_100M/    # BabyLM .train files for 100M-token corpus (optional)
├── tokenizer/
│   └── tokenizer_10M.json / tokenizer_100M.json  # BPE from train_tokenizer.py (size must match data)
├── cache/             # Cache for tokenized datasets (created automatically if needed)
├── checkpoints/       # Training checkpoints (default save_dir)
├── outputs/           # Hydra run outputs (default hydra.run.dir)
└── wandb/             # Weights & Biases logs (default wandb.save_dir)
```

Config keys (see `configs/config.yaml` and `configs/data/babylm.yaml`):

| Purpose     | Config key                   | Default path |
|------------|------------------------------|--------------|
| Data dir   | `data.data_dir`              | `{project_dir}/data/train_10M` (or `train_100M` for 100M) |
| Tokenizer  | `data.tokenizer_name_or_path`| `{project_dir}/tokenizer/tokenizer_${dataset_size}.json` |
| Cache      | `data.cache_dir`             | `{project_dir}/cache` |
| Checkpoints| `checkpointing.save_dir`     | under `{project_dir}/checkpoints/` |
| Outputs    | `hydra.run.dir`              | under `{project_dir}/outputs/` |
| Wandb      | `wandb.save_dir`             | `{project_dir}/wandb` |

Override `project_dir` (e.g. for Colab) with Hydra: `project_dir=/content/drive/MyDrive/babylm-edlm`, or override the individual keys above.

## W&B API key

Job scripts can load the API key from a file so it is not stored in the script or sbatch env:

```bash
if [ -f "${HOME}/.wandb_api_key" ]; then
  export WANDB_API_KEY=$(cat "${HOME}/.wandb_api_key")
fi
```

Create once: `echo "YOUR_WANDB_API_KEY" > ~/.wandb_api_key && chmod 600 ~/.wandb_api_key`.

## Example sbatch / run command

The BabyLM and tokenizer job scripts (`scripts/job_train_babylm_hopper.slurm`, `job_train_babylm_energy_hopper.slurm`, `job_train_tokenizer_*.slurm`) match `job_text8_mdlm_4gpu.slurm`: they `mkdir -p $SCRATCH/edlm`, clone **babylm-edlm-colab** into `$SCRATCH/edlm/babylm-edlm-colab` if missing, `cd` there, then run. You can submit from any directory on the login node:

```bash
sbatch path/to/job_train_babylm_hopper.slurm
```

**Branch:** Your local git branch is not used on the cluster. The scratch clone tracks whatever is on the remote (default branch on first clone). To run a specific remote branch, pass `EDLM_BRANCH` (and optionally `EDLM_REPO` / `EDLM_DIR` for a fork or different clone folder), e.g. `sbatch --export=NONE,EDLM_BRANCH=my-feature scripts/job_train_babylm_hopper.slurm`. For the EBM job, include `BABYLM_CKPT` in the same `--export=NONE,...` list.

That script sets `SCRATCH`, loads the same modules, uses the venv at `$SCRATCH/edlm/venv`, and runs `main` with `data=babylm` and Hopper-friendly paths. For a single-node, single-GPU run you can also run the same `python -u -m main ...` command interactively after loading modules and activating the venv from `$SCRATCH/edlm/babylm-edlm-colab`.

### BabyLM 10-epoch budget

BabyLM rules allow at most **10 epochs** of training. We use the full 10 epochs for the diffusion (MDLM) stage. The EBM stage does not add more passes over the data; you **choose which checkpoint** from that MDLM run to load as the backbone (e.g. `last.ckpt` for the end of training, or an earlier checkpoint). Run the MDLM job first, then submit the EBM job with `BABYLM_CKPT` set to the desired checkpoint path.

---

## BabyLM dataset (2026 strict) – Hugging Face + local layout

**Canonical training corpus (100M strict):** [BabyLM-community/BabyLM-2026-Strict](https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict) on Hugging Face (MIT). The repo ships per-domain `*.train.txt` files; this codebase materializes them under `data/train_100M` as:

- `babylm_strict_train.train` — training lines (90% of non-empty lines after shuffling with seed 42)
- `babylm_strict_validation.val` — held-out validation (10%)

**Tokenizer training** uses only `*.train` files, so the BPE model does not see validation lines.

**MDLM training** (`data=babylm`, `dataset_size=100M`): the dataloader calls `babylm_hf.ensure_babylm_train_files` before loading. On a cold `train_100M/`, it downloads the Hub snapshot (first run needs network; optional `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` if your environment requires auth). Hub cache defaults under `HF_HOME` or `~/.cache/huggingface`; snapshots also go under `{cache_dir}/hf_hub/...` when `data.cache_dir` is set.

**Validation:** There is no separate Hub “validation” split for this training corpus. The **validation** split used for metrics is the **local holdout** above (or, if you only have legacy `*.train` files and no `*.val`, an in-memory 90/10 split with the same seed — see `get_babylm_dataset` in `dataloader.py`).

**10M (strict-small):** There is **no** default Hugging Face download in this repo. Populate `data/train_10M/*.train` yourself (official strict-small release, symlink, or subsample from `train_100M`), then run `train_tokenizer.py --size 10M` and training with `dataset_size=10M`.

**Ordering (100M from scratch):**  
1. `sbatch scripts/job_train_tokenizer_100m.slurm` (downloads + BPE)  
2. `sbatch scripts/job_train_babylm_hopper.slurm` (`dataset_size=100M`).  

See `agent-tasks/BABYLM_2026_HF_DOWNLOAD_PLAN.md` for design notes.
