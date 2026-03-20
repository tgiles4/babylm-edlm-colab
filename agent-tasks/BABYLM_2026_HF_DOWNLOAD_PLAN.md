# Plan: BabyLM 2026 (HF) + local `.train` layout for `train_tokenizer.py` and `dataloader.py`

**Goal:** On a **fresh environment with no pre-downloaded BabyLM data**, training the BPE tokenizer (`train_tokenizer.py` via `scripts/job_train_tokenizer_100m.slurm`) and MDLM training (`scripts/job_train_babylm_hopper.slurm` with `data=babylm`) should **both work** for the **strict / 100M** track using the Hub.

**Canonical Hub dataset:** [BabyLM-community/BabyLM-2026-Strict](https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict) (MIT; [paper](https://arxiv.org/abs/2602.20092)).

**Scope note:** The Hub listing does **not** expose a separate **strict-small / 10M** config alongside **strict** in a way we can rely on for automation. **Implementation priority is strict + 100M.** The design below keeps **flexibility for `10M` / `train_10M`** (manual data, future Hub subset, or local subsample) without assuming HF download for that size.

This plan **supersedes** the older `agent-tasks/BABYLM_DOWNLOAD_PLAN.md` assumption of `cambridge-climb/BabyLM` — the 2026 release is the target.

---

## 1. Constraints inherited from current code

- **`get_babylm_dataset(data_dir, cache_dir)`** ([`dataloader.py`](../dataloader.py)) expects `data_dir` to contain at least one **`*.train`** file; each non-empty line is one training example (`text` field) after load.
- **`train_bpe_tokenizer`** ([`train_tokenizer.py`](../train_tokenizer.py)) globs **`*.train`** under `data_dir` and passes file paths to the `tokenizers` trainer.
- **Hydra** ([`configs/data/babylm.yaml`](../configs/data/babylm.yaml)) sets `data_dir: ${project_dir}/data/train_${dataset_size}` and tokenizer path `tokenizer_${dataset_size}.json`.

**On-disk layout (unchanged basename pattern):** `project_dir/data/train_100M/*.train` for the primary workflow; `train_10M/` remains a valid path when data is supplied outside HF (see §7).

---

## 2. No official validation split on Hub — we create our own

The 2026 Strict release is a **training corpus**; there is typically **no separate `validation` split** on Hugging Face suitable for MDLM eval in the same way as GLUE-style tasks. **We define validation locally.**

**Recommended policy (to document in code + `HOPPER_SETUP.md`):**

1. **After** materializing the full strict corpus into line-level examples (from HF rows or `*.train` files), build a **deterministic train/val split** with a **fixed seed** (today: `train_test_split(..., test_size=0.1, seed=42)` in `get_babylm_dataset`). This is already the behavior once `.train` files are present; the plan is to treat it explicitly as **“our val set”** — not downloaded, **held out from the same pool** as training.

2. **Optional hardening (implementation choice):**  
   - **A (minimal):** Keep current behavior: glob all `*.train`, concatenate lines, single 90/10 split in memory, cache `DatasetDict` to disk. No new files.  
   - **B (explicit files):** After download, write **separate** materialized splits, e.g. `all.train` → split into `train_shard.train` + `heldout.val` (or a `validation/` directory with `*.val`), and adjust the loader to read train vs val from those paths. Improves **inspectability** and allows **tokenizer training on train lines only** (see §5).

3. **Tokenizer vs leakage:** For BPE training, prefer **(B)** or a **train-only export** so `train_tokenizer.py` does not encode n-grams from lines that will appear only in validation. If sticking with **(A)**, document that the tokenizer may see all lines in `*.train` before the 90/10 split (current risk) or add a dedicated **train-only glob** (e.g. only `train_*.train` if split files exist).

---

## 3. Hugging Face API: verify before coding (100M / strict)

**Primary spike:**

1. `datasets.load_dataset("BabyLM-community/BabyLM-2026-Strict", ...)` — list **configs/subsets** and **splits**. Confirm the **strict / 100M** track maps to a single config (e.g. default or `strict`) and how `text` (or equivalent) is stored.
2. Confirm whether data arrives as **parquet/streaming rows** vs **per-domain `*.train.txt`** in the repo — pick one materialization path (see §4).
3. Document row/word counts vs challenge rules; do not silently assume 2023 BabyLM semantics.

**10M (optional):** If no HF config exists, **`ensure_babylm_train_files` for `10M`** should **not** call HF by default: require **existing `*.train`** under `data/train_10M` or document a **manual** workflow (copy, symlink, or subsample from `train_100M`). Leave a **placeholder branch** (e.g. `if hf_config_for_10m:`) if a Hub subset appears later.

Record exact `load_dataset` signatures in a module comment at the top of the helper.

---

## 4. Shared helper: download + materialize (single source of truth)

**Add a small module** (recommended: `babylm_hf.py` at repo root) to avoid `train_tokenizer.py` importing all of `dataloader.py`:

**Function (name illustrative):** `ensure_babylm_train_files(data_dir, dataset_size, hf_cache_dir=None) -> None`

**Behavior:**

1. **100M:** Map to Hub **strict** (exact config name from §3 spike). Download/materialize into `data_dir` as one or more **`*.train`** files (line-per-example), with locking and idempotency as in the original plan.
2. **10M:** **Default — no HF download.** If `data_dir` has no `*.train`, raise a **clear error** pointing to: place files manually, subsample from 100M, or enable a future flag when HF exposes 10M. If `*.train` already exist, succeed (supports local strict-small–equivalent experiments).
3. **Idempotency, locks, caches, `HF_TOKEN`:** unchanged from prior plan (see old §3 in git history if needed).

---

## 5. Train/val creation and tokenizer script

- **`get_babylm_dataset`:** Continue to expose **`train` / `validation`** via the chosen split policy (§2). If moving to **explicit `.val` files**, update this function to load train and val from separate globs or paths; keep **`cache_dir`** invalidation rules coherent when split logic changes.

- **`train_tokenizer.py`:**  
  - For **100M**, call `ensure_babylm_train_files` then run BPE on **train-only** lines if splits are materialized (§2B); otherwise document leakage caveat (§2A).  
  - **`--size 10M`:** unchanged CLI; works when `data/train_10M` is populated manually or via future download.

---

## 6. Changes to `dataloader.py` (summary)

1. **`get_babylm_dataset`:** Before `glob('*.train')`, call **`ensure_babylm_train_files`** for **100M**; for **10M**, only call if/when HF support exists, else rely on existing files. Add **`dataset_size`** or infer from `data_dir` basename (`train_100M` → `100M`).
2. **`get_dataset`:** Pass **`dataset_size`** from config for `babylm`.
3. If implementing **§2B**, add parameters or conventions for **val file paths** (`*.val` or `data_dir` subdirs).

---

## 7. Config / `10M` flexibility

- [`configs/config.yaml`](../configs/config.yaml) defaults `dataset_size: 10M`; **Hopper strict-100M runs** should use **`dataset_size=100M`** and `data.data_dir` pointing at `train_100M` (already templated in `babylm.yaml`).
- **10M without HF:** Users set `dataset_size=10M` and populate `data/train_10M` themselves; tokenizer and training use the same path patterns as today.

---

## 8. Slurm scripts

| Script | Role | Plan |
|--------|------|------|
| [`scripts/job_train_tokenizer_100m.slurm`](../scripts/job_train_tokenizer_100m.slurm) | CPU; `--size 100M` | Network for HF on first run; walltime for download + BPE. |
| [`scripts/job_train_tokenizer_10m.slurm`](../scripts/job_train_tokenizer_10m.slurm) | `--size 10M` | Document that **data must exist locally** unless future HF 10M is added; same machinery once `train_10M` has `*.train`. |
| [`scripts/job_train_babylm_hopper.slurm`](../scripts/job_train_babylm_hopper.slurm) | `data=babylm` | Use **`dataset_size=100M`** for strict-100M workflow; val comes from **local split** (§2), not Hub. |

**Ordering (scratch, 100M):** tokenizer job (downloads + BPE) → training job (`dataset_size=100M`).

---

## 9. Docs to update

- **`HOPPER_SETUP.md`:** Strict 100M from HF; **validation = local holdout**; optional `.val` files if implemented.
- **`TOKENIZER_GUIDE.md` / `colab/COLAB.md`:** 100M primary; 10M = bring-your-own data until Hub supports it.
- **Banner** on `agent-tasks/BABYLM_DOWNLOAD_PLAN.md` → point here.

---

## 10. Testing checklist

- [ ] Cold start **100M:** empty `data/train_100M` → HF download → `*.train` → tokenizer writes `tokenizer_100M.json`.
- [ ] **Validation:** `get_babylm_dataset` yields non-empty `validation` split; metrics use that split (document seed/ratio).
- [ ] Repeat run: idempotent download / materialize.
- [ ] **10M:** With manually placed `train_10M/*.train`, tokenizer + `main.py data=babylm dataset_size=10M` runs without HF.
- [ ] **`--no_download`:** fails clearly if `*.train` missing.

---

## 11. Summary

| Component | Action |
|-----------|--------|
| Hub source | [BabyLM-2026-Strict](https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict), **strict / 100M** only for automated download |
| Validation | **Not** from HF; **local** holdout (existing 90/10 or explicit `*.val` / split files) |
| 10M | **Flexible:** manual/future HF; no default HF download until a subset exists |
| `train_tokenizer.py` / `get_babylm_dataset` | `ensure_babylm_train_files`; split policy + optional tokenizer-on-train-only |

This aligns implementation with **strict + 100M** as the supported zero-to-run path, documents **our val set**, and preserves **`train_10M` / `tokenizer_10M`** for experiments without requiring a Hub 10M dataset today.
