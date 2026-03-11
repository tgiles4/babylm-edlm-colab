# Hopper cleanup – agent todo list

**Goal:** Clean up this codebase to run on GMU's supercomputer cluster Hopper. Hopper-first defaults; keep Colab as an optional path.

**Reference scripts:** `scripts/job_text8_mdlm_4gpu.slurm` and `scripts/job_text8_nce_4gpu.slurm` (parent codebase). Use them for:
- Environment: `SCRATCH`, `path=${SCRATCH}/edlm`, venv at `$SCRATCH/edlm/venv`, Python 3.9, CUDA 12.4, PyTorch 2.6
- Modules: `hosts/hopper`, `gnu10/10.3.0-ya`, `python/3.9.9-jh`, `cuda/12.4.0`
- Job structure: `#SBATCH` (qos=gpu, partition=contrib-gpuq, gres=gpu:A100.80gb:N), `--export=NONE`, set SCRATCH in script
- Paths: `checkpointing.save_dir=${path}`, `hydra.run.dir=outputs/<name>`, WANDB_API_KEY from `~/.wandb_api_key`
- Run: `"$ENV/bin/python" -u -m main` with Hydra overrides; multi-GPU use `srun --ntasks=N --export=ALL`

---

## 1. Config defaults for Hopper

- [ ] **configs/config.yaml**
  - Replace all `/content/drive/MyDrive/babylm-edlm/...` with Hopper-friendly defaults using `$SCRATCH` or `$WORK`/`$PROJECT` (e.g. `${SCRATCH}/babylm-edlm/...` or env-substituted paths). If Hydra does not expand shell env vars by default, use a single base key (e.g. `project_dir`) that defaults to `$SCRATCH/babylm-edlm` and compose paths under it, or document that users must set env and override.
  - Update: `wandb.save_dir`, `hydra.run.dir`, `checkpointing.save_dir` so default run works on Hopper without overrides.
  - Keep comments that mention Colab overrides where useful.

- [ ] **configs/data/babylm.yaml**
  - Replace Colab paths with Hopper defaults for `tokenizer_name_or_path`, `cache_dir`, `data_dir` (e.g. `$SCRATCH/babylm-edlm/tokenizer/tokenizer.json`, `$SCRATCH/babylm-edlm/cache`, `$SCRATCH/babylm-edlm/data/train_10M`). Use the same convention as config.yaml (env var or single base path). Document in HOPPER_SETUP.md where to place data and tokenizer on the cluster.

---

## 2. Colab preserved but isolated

- [ ] Move Colab-only files into **colab/** subfolder:
  - `install_colab.sh`
  - `install_colab.py`
  - `install_colab_conda.sh`
  - `install_colab_conda.py`
  - `colab_train_cell.py`
  - Update any in-repo references to these (e.g. README) to point to `colab/`.

- [ ] Add **colab/COLAB.md** that explains:
  - Run the install script from the `colab/` folder (or document path to it).
  - Required Hydra overrides for Colab: `data.data_dir`, `data.tokenizer_name_or_path`, `data.cache_dir`, `checkpointing.save_dir`, `hydra.run.dir`, and wandb `save_dir` (and project output dir if different). Use example Colab/Drive paths so users can copy-paste.

- [ ] Optionally add a short **"Colab (optional)"** section in the main **README.md** linking to `colab/COLAB.md`.

---

## 3. Docs

- [ ] Add **HOPPER_SETUP.md** (or a cluster section in README) with:
  - **Module load:** `module load hosts/hopper gnu10/10.3.0-ya python/3.9.9-jh cuda/12.4.0` (match example scripts).
  - **Conda/venv:** Environment name or path (e.g. venv at `$SCRATCH/edlm/venv` with PyTorch 2.6).
  - **Where data/checkpoints live:** e.g. `$SCRATCH/babylm-edlm/data/`, `$SCRATCH/babylm-edlm/checkpoints/`, tokenizer path; note `--export=NONE` and setting `SCRATCH` in the job script if needed.
  - **One example sbatch/run command:** e.g. run the new BabyLM Hopper script from task 4, or `sbatch scripts/job_train_babylm_hopper.slurm`.
  - Structure and conventions should match `scripts/job_text8_mdlm_4gpu.slurm` / `job_text8_nce_4gpu.slurm`.

- [ ] **Consolidate Colab docs:** Merge or trim so Colab is one optional path:
  - Keep one Colab entry point: `colab/COLAB.md` (and install scripts in `colab/`).
  - Remove or redirect from repo root: redundant Colab-focused docs (e.g. multiple install/setup guides at root that duplicate `colab/COLAB.md`). Ensure main README or docs point to `colab/COLAB.md` for Colab-only steps rather than maintaining several Colab docs at root.

---

## 4. Scripts

- [ ] Add a **Hopper SLURM job script** for BabyLM (e.g. **scripts/job_train_babylm_hopper.slurm** or **scripts/slurm_train_babylm.sh**):
  - Match structure of `job_text8_mdlm_4gpu.slurm`: set `SCRATCH`, `path`, clone or use repo, load same modules, use same venv (`ENV=$SCRATCH/edlm/venv` or equivalent).
  - Set project path and `cd` into this repo (or clone this repo if different from Energy-Diffusion-LLM).
  - Run `main.py` with `data=babylm` and Hopper-specific overrides: `checkpointing.save_dir`, `hydra.run.dir`, and data/tokenizer/cache paths consistent with config defaults from task 1. Use `"$ENV/bin/python" -u -m main` and Hydra `++path=${path}` if needed.
  - No Colab or Google Drive paths. Optionally support 1 GPU (like text8_mdlm) or 2+ GPUs with `srun` (like text8_nce) and `loader.num_workers=0` for multi-GPU if needed.
  - Reference existing `scripts/job_train_owt_ebm.sh` for `main.py` flags (e.g. `use_energy`, `noise=cosine`) but keep paths and job layout from the SLURM examples.

---

## 5. Tokenizer script

- [ ] In **train_tokenizer.py**:
  - Change example paths in module docstring and argparse help to Hopper or generic (e.g. `$SCRATCH/babylm-edlm/data/train_10M`, `$SCRATCH/babylm-edlm/tokenizer/tokenizer.json`).
  - Remove or minimize Colab-specific example paths from the script itself; document Colab paths only in **colab/COLAB.md**.

---

## Do not

- Do not remove the core Colab install or run instructions; keep them in **colab/** so Colab remains usable via path overrides.
