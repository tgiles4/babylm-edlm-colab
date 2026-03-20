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
│   └── tokenizer.json # BPE tokenizer (train with train_tokenizer.py on the .train data)
├── cache/             # Cache for tokenized datasets (created automatically if needed)
├── checkpoints/       # Training checkpoints (default save_dir)
├── outputs/           # Hydra run outputs (default hydra.run.dir)
└── wandb/             # Weights & Biases logs (default wandb.save_dir)
```

Config keys (see `configs/config.yaml` and `configs/data/babylm.yaml`):

| Purpose     | Config key                   | Default path |
|------------|------------------------------|--------------|
| Data dir   | `data.data_dir`              | `{project_dir}/data/train_10M` (or `train_100M` for 100M) |
| Tokenizer  | `data.tokenizer_name_or_path`| `{project_dir}/tokenizer/tokenizer.json` |
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

## BabyLM dataset – access patterns (no download code)

You need the **BabyLM challenge** data as **plain-text `.train` files** (one sentence per line). The training code expects a directory of `*.train` files (e.g. `bnc_spoken.train`, `childes.train`, `gutenberg.train`, etc.); it discovers all `*.train` in `data_dir` and uses them.

### Where the data comes from

- **Official source:** BabyLM Challenge (babylm.github.io or the challenge organizers’ repository). The benchmark provides **strict** and **strict-small** evaluation sets; for **training**, use the released training corpora that match the 10M and 100M token settings.
- **Hugging Face:** The BabyLM datasets may be hosted on Hugging Face (e.g. under a `babylm` or challenge organization). Search for “BabyLM” or “babylm strict” to find the correct dataset card and files.
- **Format:** You need the **raw training text** (or pre-split `.train` files). The pipeline expects **one sentence per line** in `.train` files; if the official release is in another format (e.g. JSON, single file), you must **convert or split** it into one `.train` file per source (or one combined `.train` file) and place them in `data/train_10M` or `data/train_100M`.

### How to get and place the data (pattern only)

1. **Obtain:** From the official BabyLM site or Hugging Face, download (or clone) the training data for the 10M and/or 100M token regime. Prefer the official challenge data so results are comparable.
2. **Format:** Ensure the data is (or is converted to) one sentence per line, in files with extension `.train`.
3. **Place:** Put 10M-corpus `.train` files under `$SCRATCH/babylm-edlm/data/train_10M` and, if using 100M, under `$SCRATCH/babylm-edlm/data/train_100M`. The code uses `data.data_dir` (default `{project_dir}/data/train_10M`).
4. **Tokenizer:** After the `.train` files are in place, train a BPE tokenizer with `train_tokenizer.py` (see its help and repo docs), writing `tokenizer.json` to `$SCRATCH/babylm-edlm/tokenizer/tokenizer.json`, or override `data.tokenizer_name_or_path` to point to your tokenizer path.

No download or conversion scripts are provided here; use the official or Hugging Face access methods (browser, CLI, or your own scripts) and follow the layout above.
