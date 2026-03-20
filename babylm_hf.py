"""BabyLM 2026 (HF) materialization for local ``*.train`` / ``*.val`` layout.

Canonical Hub dataset: ``BabyLM-community/BabyLM-2026-Strict``
https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict

The repository hosts per-domain plain-text ``*.train.txt`` files (e.g. ``bnc_spoken.train.txt``).
You may also load via ``datasets`` (builder/parquet layout can change between revisions)::

  datasets.load_dataset("BabyLM-community/BabyLM-2026-Strict", split="train")

This module uses ``huggingface_hub.snapshot_download`` so materialization does not depend on a
specific ``datasets`` builder or parquet conversion.

``10M`` / strict-small: there is no automated Hub download here; supply ``*.train`` under
``data/train_10M`` manually or subsample from ``train_100M``. A future Hub subset could be wired
behind an explicit flag (``if hf_config_for_10m:``).
"""

from __future__ import annotations

import glob
import os
import random
import time
import typing

import utils

LOGGER = utils.get_logger(__name__)

BABYLM_2026_STRICT_REPO = "BabyLM-community/BabyLM-2026-Strict"
SPLIT_SEED = 42
TRAIN_FRACTION = 0.9
MARKER_NAME = ".babylm_2026_strict_ok"
TRAIN_OUT = "babylm_strict_train.train"
VAL_OUT = "babylm_strict_validation.val"
LOCK_NAME = ".babylm_2026_download.lock"
DOWNLOAD_WAIT_S = 7200


def infer_dataset_size_from_data_dir(data_dir: str) -> typing.Optional[str]:
  base = os.path.basename(os.path.normpath(data_dir))
  if "100M" in base:
    return "100M"
  if "10M" in base:
    return "10M"
  return None


def _has_train_files(data_dir: str) -> bool:
  return bool(glob.glob(os.path.join(data_dir, "*.train")))


def _snapshot_has_train_txt(src_dir: str) -> bool:
  return bool(glob.glob(os.path.join(src_dir, "*.train.txt")))


def _hub_token() -> typing.Optional[str]:
  return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _materialize_from_txt_sources(src_dir: str, data_dir: str) -> None:
  txt_files = sorted(glob.glob(os.path.join(src_dir, "*.train.txt")))
  if not txt_files:
    raise RuntimeError(
      f"No *.train.txt files under {src_dir} after download. "
      "The BabyLM-2026-Strict dataset layout may have changed.")
  lines: typing.List[str] = []
  for path in txt_files:
    LOGGER.info(f"Reading {os.path.basename(path)}...")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
      for line in f:
        line = line.strip()
        if line:
          lines.append(line)
  LOGGER.info(f"Collected {len(lines)} non-empty lines; shuffling (seed={SPLIT_SEED})...")
  rng = random.Random(SPLIT_SEED)
  rng.shuffle(lines)
  n_train = int(len(lines) * TRAIN_FRACTION)
  train_lines = lines[:n_train]
  val_lines = lines[n_train:]
  os.makedirs(data_dir, exist_ok=True)
  train_path = os.path.join(data_dir, TRAIN_OUT)
  val_path = os.path.join(data_dir, VAL_OUT)
  with open(train_path, "w", encoding="utf-8") as ft:
    for t in train_lines:
      ft.write(t + "\n")
  with open(val_path, "w", encoding="utf-8") as fv:
    for t in val_lines:
      fv.write(t + "\n")
  marker = os.path.join(data_dir, MARKER_NAME)
  with open(marker, "w", encoding="utf-8") as m:
    m.write(BABYLM_2026_STRICT_REPO + "\n")
  LOGGER.info(
    f"Wrote {train_path} ({len(train_lines)} lines) and {val_path} ({len(val_lines)} lines)")


def _default_hf_cache(hf_cache_dir: typing.Optional[str]) -> str:
  if hf_cache_dir:
    return hf_cache_dir
  return os.environ.get("HF_HOME") or os.path.expanduser(
    os.path.join("~", ".cache", "huggingface"))


def ensure_babylm_train_files(
    data_dir: str,
    dataset_size: str,
    hf_cache_dir: typing.Optional[str] = None,
    no_download: bool = False,
) -> None:
  """Ensure ``data_dir`` contains BabyLM training text for tokenizer / dataloader.

  * **100M:** If there are no ``*.train`` files, download the 2026 Strict corpus from the Hub,
    then write ``babylm_strict_train.train`` and ``babylm_strict_validation.val`` (90/10, seed 42).
    BPE training uses only ``*.train`` (no val leakage into the tokenizer).
  * **10M:** No Hub download. If ``*.train`` is missing, raise with instructions.

  Idempotent: if any ``*.train`` file already exists, returns without downloading.
  """
  os.makedirs(data_dir, exist_ok=True)
  if _has_train_files(data_dir):
    return

  if dataset_size == "10M":
    if no_download:
      raise FileNotFoundError(
        f"No *.train files in {data_dir} and --no_download was set.\n"
        "For 10M, place strict-small ``*.train`` files under this directory, or subsample from "
        "train_100M; automated Hub download is only implemented for 100M (BabyLM-2026-Strict).")
    raise ValueError(
      f"No *.train files in {data_dir}.\n"
      "BabyLM 10M (strict-small): add ``*.train`` files locally (official release or subsample "
      "from ``train_100M``). There is no default Hugging Face auto-download for 10M until a Hub "
      "subset is published. See agent-tasks/BABYLM_2026_HF_DOWNLOAD_PLAN.md.")

  if dataset_size != "100M":
    raise ValueError(f"Unknown dataset_size={dataset_size!r}; expected '10M' or '100M'.")

  if no_download:
    raise FileNotFoundError(
      f"No *.train files in {data_dir} and --no_download was set.\n"
      "Run without --no_download to fetch BabyLM-2026-Strict from the Hub, or place "
      "``*.train`` files manually.")

  # if hf_config_for_10m:
  #   ... optional future Hub subset for strict-small ...

  lock_path = os.path.join(data_dir, LOCK_NAME)
  builder = False
  try:
    open(lock_path, "x").close()
    builder = True
  except FileExistsError:
    LOGGER.info(f"Waiting for BabyLM download lock at {lock_path}...")
    deadline = time.monotonic() + DOWNLOAD_WAIT_S
    while not _has_train_files(data_dir):
      if time.monotonic() > deadline:
        try:
          os.remove(lock_path)
        except OSError:
          pass
        raise TimeoutError(
          f"BabyLM data in {data_dir} did not appear within {DOWNLOAD_WAIT_S}s.")
      time.sleep(2)
    return

  try:
    from huggingface_hub import snapshot_download

    cache_base = _default_hf_cache(hf_cache_dir)
    snap_dir = os.path.join(cache_base, "babylm_edlm_snapshots", "BabyLM-2026-Strict")
    if not _snapshot_has_train_txt(snap_dir):
      LOGGER.info(
        f"Downloading {BABYLM_2026_STRICT_REPO} to {snap_dir} (first run may take a while)...")
      os.makedirs(os.path.dirname(snap_dir), exist_ok=True)
      snapshot_download(
        repo_id=BABYLM_2026_STRICT_REPO,
        repo_type="dataset",
        local_dir=snap_dir,
        local_dir_use_symlinks=False,
        token=_hub_token(),
      )
    _materialize_from_txt_sources(snap_dir, data_dir)
  finally:
    try:
      os.remove(lock_path)
    except OSError:
      pass
