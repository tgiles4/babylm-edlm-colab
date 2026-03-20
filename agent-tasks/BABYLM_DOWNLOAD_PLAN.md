# Plan: Download BabyLM 10M or 100M (Same Pattern as Text8)

> **Superseded for the 2026 challenge:** Use **`BABYLM_2026_HF_DOWNLOAD_PLAN.md`** — Hub source [BabyLM-community/BabyLM-2026-Strict](https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict), implementation in `babylm_hf.py`. The notes below referred to the older `cambridge-climb/BabyLM` layout.

**Goal:** Obtain the 10M or 100M BabyLM training data automatically when the dataset is first requested, using the same pattern as Text8 and other datasets in this codebase—no separate download scripts.

---

## 1. Pattern to follow (Text8 / others)

- **Text8:** `get_text8_dataset(cache_dir, ...)` in `dataloader.py` checks for existing processed data; if missing, checks for raw files; if missing, downloads from `http://mattmahoney.net/dc/text8.zip`, splits, and saves under `cache_dir`. All logic lives in the dataloader.
- **OpenWebText, WikiText, etc.:** `get_dataset()` calls `datasets.load_dataset(...)` with `cache_dir`; the library downloads and caches on first use.
- **BabyLM (desired):** Same idea—when `data=babylm` is used and `get_babylm_dataset(data_dir, ...)` is called, if `data_dir` is missing or contains no `*.train` (and optionally no `*.txt`) files, **download from Hugging Face into `data_dir`**, then load. No separate script; first run triggers the download.

---

## 2. Source of truth

- **Dataset:** [cambridge-climb/BabyLM](https://huggingface.co/datasets/cambridge-climb/BabyLM) on Hugging Face (official 2023 BabyLM Challenge data).
- **Layout on HF:**
  - `clean/10M/` — 10M-word training set (strict-small): 10 `.txt` files (one per domain).
  - `clean/100M/` — 100M-word training set (strict): same 10 filenames, larger content.
  - Domains: `aochildes.txt`, `bnc_spoken.txt`, `cbt.txt`, `children_stories.txt`, `gutenberg.txt`, `open_subtitles.txt`, `qed.txt`, `simple_wikipedia.txt`, `switchboard.txt`, `wikipedia.txt`.
- **Format:** Plain text, newline-separated lines. Compatible with “one line per example” in `get_babylm_dataset()`.

---

## 3. Where the data lands

- **10M:** `data_dir` when it points at a path like `{project_dir}/data/train_10M` (config default).
- **100M:** `data_dir` when user overrides with e.g. `data.data_dir=${project_dir}/data/train_100M`.
- **Which size to download:** Infer from `data_dir`: if the path ends with `train_100M` (or contains `100M`), fetch `clean/100M/`; otherwise fetch `clean/10M/`. Alternatively, pass an explicit size from config (e.g. `dataset_size` or a new `data.babylm_size`) into the loader so download and config stay in sync.

---

## 4. Download inside the dataloader

- **Where:** In `dataloader.py`, either:
  - **Option A:** At the start of `get_babylm_dataset(data_dir, cache_dir=None)`, if `data_dir` has no `*.train` (and no `*.txt` if we support that), call a small helper that downloads from HF into `data_dir`, then continue with the existing file-based load; or
  - **Option B:** In `get_dataset()` when `dataset_name == 'babylm'`, before calling `get_babylm_dataset()`, ensure `data_dir` is populated (same “if empty, download” logic), then call `get_babylm_dataset()` as today.

- **How to download:** Use **Hugging Face Hub** so behavior matches the rest of the stack:
  - **`huggingface_hub`:** `hf_hub_download` (per file) or `snapshot_download(repo_id="cambridge-climb/BabyLM", allow_patterns="clean/10M/*" or "clean/100M/*")` to get the right subset, then move/copy files from the cache into `data_dir` and rename `.txt` → `.train`; or
  - **`datasets.load_dataset`:** If the dataset exposes 10M and 100M train splits clearly, load the appropriate split and write one `.train` file per domain (or one combined file) into `data_dir`. Prefer the approach that avoids reimplementing the HF dataset’s custom logic in `BabyLM.py`.

- **Idempotency:** If `data_dir` already contains the expected `*.train` files (e.g. at least one, or the full list of 10), skip download and proceed with the current load logic.

---

## 5. File mapping (HF → our layout)

When writing into `data_dir`, save HF’s `.txt` content as `.train` so existing `get_babylm_dataset()` (which globs `*.train`) works unchanged:

| HF file (clean/10M or clean/100M) | Our file in data_dir |
|-----------------------------------|----------------------|
| aochildes.txt                     | aochildes.train      |
| bnc_spoken.txt                    | bnc_spoken.train     |
| cbt.txt                           | cbt.train            |
| children_stories.txt              | children_stories.train |
| gutenberg.txt                     | gutenberg.train      |
| open_subtitles.txt                | open_subtitles.train |
| qed.txt                           | qed.train            |
| simple_wikipedia.txt              | simple_wikipedia.train |
| switchboard.txt                   | switchboard.train    |
| wikipedia.txt                     | wikipedia.train      |

---

## 6. Config / API

- **configs/data/babylm.yaml:** Keeps `data_dir: ${project_dir}/data/train_10M`. For 100M, user sets `data.data_dir=${project_dir}/data/train_100M` (and typically `dataset_size=100M`). No change required beyond the dataloader.
- **Optional:** If we infer size from path, we can document that `.../train_10M` triggers 10M download and `.../train_100M` triggers 100M download. No new config keys strictly necessary.

---

## 7. Docs to update

- **HOPPER_SETUP.md:** Replace “run a download script” with: BabyLM is **downloaded automatically on first use** when using `data=babylm`, like Text8. Use `data_dir` default for 10M; override `data.data_dir=.../train_100M` for 100M.
- **README / colab/COLAB.md:** State that 10M and 100M are both supported; first run with the corresponding `data_dir` will download from Hugging Face if the directory is empty.

---

## 8. Summary

| Step | Action |
|------|--------|
| 1 | In `dataloader.py`, before loading BabyLM from disk, check if `data_dir` is empty (no `*.train` files). If empty, download from `cambridge-climb/BabyLM` (subdir `clean/10M` or `clean/100M` based on `data_dir` or config). |
| 2 | Use `huggingface_hub` or `datasets` to fetch the chosen subset; write files into `data_dir` as `*.train` (see mapping table). |
| 3 | Proceed with existing `get_babylm_dataset()` logic (glob `*.train`, combine, 90/10 split). No separate download script. |
| 4 | Update HOPPER_SETUP.md and any Colab/README text to say BabyLM is auto-downloaded on first use, with 10M vs 100M determined by `data_dir` (or config). |

This keeps BabyLM consistent with Text8 and the other datasets: download is part of the dataloader, and users get either 10M or 100M by setting `data_dir` (and optionally `dataset_size`) accordingly.
