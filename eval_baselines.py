"""Evaluate external HuggingFace baselines on the BabyLM validation set.

Computes NLL, PPL, and BPD for four model families:

  1. despoinakk/diffusion_cosine_babylm  (diffusion ELBO, 28 checkpoints)
  2. BabyLM-community/babylm-baseline-100m-gpt2  (causal cross-entropy)
  3. BabyLM-community/babylm-baseline-100m-gpt-bert-masked-focus  (3 methods)
  4. BabyLM-community/babylm-baseline-100m-gpt-bert-causal-focus  (3 methods)

Usage examples
--------------
Diffusion (one checkpoint):
    python eval_baselines.py --model_type diffusion \\
        --model_name_or_path despoinakk/diffusion_cosine_babylm \\
        --checkpoint_subfolder chck_100M \\
        --data_dir $SCRATCH/babylm-edlm/data/train_100M \\
        --output_json results/diffusion_100M.json

GPT-2:
    python eval_baselines.py --model_type ar \\
        --model_name_or_path BabyLM-community/babylm-baseline-100m-gpt2 \\
        --data_dir $SCRATCH/babylm-edlm/data/train_100M \\
        --output_json results/gpt2.json

GPT-BERT:
    python eval_baselines.py --model_type gpt_bert \\
        --model_name_or_path BabyLM-community/babylm-baseline-100m-gpt-bert-masked-focus \\
        --data_dir $SCRATCH/babylm-edlm/data/train_100M \\
        --methods causal,masked_ce,pll \\
        --output_json results/gpt_bert_masked.json
"""

import argparse
import glob
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
)

LOG2 = math.log(2)
NEG_INF = -1_000_000.0


# ---------------------------------------------------------------------------
# Metrics (mirrored from diffusion.py for standalone use)
# ---------------------------------------------------------------------------

class NLL(torchmetrics.aggregation.MeanMetric):
    pass


class BPD(NLL):
    def compute(self):
        return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
    def compute(self):
        return torch.exp(self.mean_value / self.weight)


def create_metrics(device="cpu"):
    metrics = torchmetrics.MetricCollection({
        "nll": NLL(),
        "bpd": BPD(),
        "ppl": Perplexity(),
    })
    metrics.set_dtype(torch.float64)
    return metrics.to(device)


# ---------------------------------------------------------------------------
# Cosine noise schedule (mirrored from noise_schedule.py)
# ---------------------------------------------------------------------------

class CosineNoise(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, t):
        cos = torch.cos(t * torch.pi / 2)
        sigma = -torch.log(self.eps + (1 - self.eps) * cos)
        sin = (1 - self.eps) * torch.sin(t * torch.pi / 2)
        dsigma = (torch.pi / 2) * sin / ((1 - self.eps) * cos + self.eps)
        return sigma, dsigma


# ---------------------------------------------------------------------------
# Validation-text loading (mirrors dataloader.get_babylm_dataset split logic)
# ---------------------------------------------------------------------------

def load_val_texts(data_dir):
    """Return a list of validation-split strings from *data_dir*.

    If ``*.val`` files exist they are used directly; otherwise the ``*.train``
    files are combined and split 90/10 with ``seed=42`` (matching the
    codebase's ``get_babylm_dataset``).
    """
    val_files = sorted(glob.glob(os.path.join(data_dir, "*.val")))
    if val_files:
        texts = []
        for vf in val_files:
            with open(vf, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append(line)
        print(f"Loaded {len(texts)} val texts from {len(val_files)} .val files")
        return texts

    train_files = sorted(glob.glob(os.path.join(data_dir, "*.train")))
    if not train_files:
        raise FileNotFoundError(
            f"No .train or .val files found in {data_dir}")
    all_texts = []
    for tf in train_files:
        with open(tf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_texts.append(line)
    rng = random.Random(42)
    rng.shuffle(all_texts)
    n_train = int(len(all_texts) * 0.9)
    val_texts = all_texts[n_train:]
    print(f"Loaded {len(val_texts)} val texts via 90/10 split of "
          f"{len(train_files)} .train files (seed=42)")
    return val_texts


# ---------------------------------------------------------------------------
# Simple tokenized dataset / dataloader
# ---------------------------------------------------------------------------

class TokenizedDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512,
                 tok_batch_size=10_000):
        chunks_ids, chunks_mask = [], []
        for i in range(0, len(texts), tok_batch_size):
            enc = tokenizer(
                texts[i : i + tok_batch_size],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            chunks_ids.append(enc["input_ids"])
            chunks_mask.append(enc["attention_mask"])
        self.input_ids = torch.cat(chunks_ids)
        self.attention_mask = torch.cat(chunks_mask)

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


def make_dataloader(texts, tokenizer, max_length, batch_size):
    ds = TokenizedDataset(texts, tokenizer, max_length=max_length)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=True)


# ---------------------------------------------------------------------------
# Generic evaluation loop
# ---------------------------------------------------------------------------

def eval_loop(compute_fn, dataloader, metrics, desc="Evaluating"):
    for m in metrics.values():
        m.reset()
    n = len(dataloader)
    for i, batch in enumerate(dataloader):
        nlls, mask = compute_fn(batch)
        for m in metrics.values():
            m.update(nlls, mask)
        if (i + 1) % max(1, n // 10) == 0 or i == n - 1:
            print(f"  [{desc}] {i + 1}/{n}", flush=True)
    return {name: m.compute().item() for name, m in metrics.items()}


# ===================================================================
# 1. Diffusion ELBO evaluation (despoinakk/diffusion_cosine_babylm)
# ===================================================================

class DiffusionBERTWrapper(nn.Module):
    """Wraps the despoinakk ``BERTForMaskedLM`` to expose a
    ``forward(indices, sigma, attention_mask)`` interface suitable for
    the diffusion ELBO computation.

    The upstream HF ``forward`` hard-codes ``sigma=0``; this wrapper calls
    the inner ``Bert.get_contextualized_embeddings`` directly so that an
    arbitrary sigma schedule can be passed.
    """

    def __init__(self, hf_model):
        super().__init__()
        self.bert = hf_model.model          # inner Bert with sigma_map
        self.classifier = hf_model.lm_head   # MaskClassifier head
        self.vocab_size = hf_model.vocab_size
        self.hidden_size = hf_model.hidden_size

    @torch.no_grad()
    def forward(self, indices, sigma, attention_mask=None):
        """
        Parameters
        ----------
        indices : Tensor[B, T]  token ids (possibly noised)
        sigma   : Tensor[B]     noise-level conditioning
        attention_mask : Tensor[B, T]  1 = valid, 0 = padding

        Returns
        -------
        logits : Tensor[B, T, V]  raw (pre-softmax) logits
        """
        B, T = indices.shape
        input_ids_t = indices.transpose(0, 1)  # [T, B] expected by Bert

        if attention_mask is None:
            attn_t = torch.ones(T, B, dtype=torch.long, device=indices.device)
        else:
            attn_t = attention_mask.transpose(0, 1)

        contextualized, _ = self.bert.get_contextualized_embeddings(
            input_ids_t, attn_t, mask_p=sigma, eval_=False)

        seq_out = contextualized.transpose(0, 1).contiguous()       # [B, T, D]
        logits = self.classifier.nonlinearity(seq_out)               # [B, T, V]
        return logits


def compute_diffusion_elbo(wrapper, input_ids, attention_mask, noise,
                           mask_index, device,
                           antithetic_sampling=True, sampling_eps=1e-3):
    """ELBO NLL for one batch (subs parameterization, continuous time)."""
    x0 = input_ids.to(device)
    attn = attention_mask.to(device).float()
    B = x0.shape[0]

    # --- sample t --------------------------------------------------------
    eps_t = torch.rand(B, device=device)
    if antithetic_sampling:
        offset = torch.arange(B, device=device).float() / B
        eps_t = (eps_t / B + offset) % 1
    t = (1 - sampling_eps) * eps_t + sampling_eps

    # --- noise -----------------------------------------------------------
    sigma, dsigma = noise(t)
    move_chance = 1 - torch.exp(-sigma[:, None])

    # --- noisy input (absorbing-state corruption) ------------------------
    move = torch.rand(*x0.shape, device=device) < move_chance
    xt = torch.where(move, mask_index, x0)

    # --- forward pass ----------------------------------------------------
    logits = wrapper(xt, sigma, attention_mask.to(device))

    # --- subs parameterization  -> log p_theta ---------------------------
    logits[:, :, mask_index] += NEG_INF
    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    unmasked = (xt != mask_index)
    logits[unmasked] = NEG_INF
    logits[unmasked, xt[unmasked]] = 0

    log_p_theta = torch.gather(logits, -1, x0[:, :, None]).squeeze(-1)

    # --- ELBO loss per token ---------------------------------------------
    loss = -log_p_theta * (dsigma / torch.expm1(sigma))[:, None]
    return loss * attn, attn


def eval_diffusion(args):
    """Evaluate a single despoinakk checkpoint via diffusion ELBO."""
    subfolder = args.checkpoint_subfolder or None
    print(f"Loading diffusion model from {args.model_name_or_path}"
          f"{f' (subfolder={subfolder})' if subfolder else ''} ...")

    hf_model = AutoModelForMaskedLM.from_pretrained(
        args.model_name_or_path, subfolder=subfolder,
        trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, subfolder=subfolder)

    device = torch.device(args.device)
    hf_model.eval().to(device)
    wrapper = DiffusionBERTWrapper(hf_model)
    noise = CosineNoise(eps=1e-3).to(device)

    mask_index = (tokenizer.mask_token_id
                  if tokenizer.mask_token_id is not None
                  else tokenizer.vocab_size)
    print(f"  vocab_size={tokenizer.vocab_size}  mask_index={mask_index}")

    val_texts = load_val_texts(args.data_dir)
    dl = make_dataloader(val_texts, tokenizer, args.seq_length,
                         args.batch_size)

    metrics = create_metrics(device)

    def compute_fn(batch):
        return compute_diffusion_elbo(
            wrapper, batch["input_ids"], batch["attention_mask"],
            noise, mask_index, device)

    results = eval_loop(compute_fn, dl, metrics, desc="Diffusion ELBO")
    results["method"] = "diffusion_elbo"
    return results


# ===================================================================
# 2. Autoregressive evaluation (GPT-2 baseline)
# ===================================================================

def compute_causal_nll(model, input_ids, attention_mask, device):
    """Standard next-token-prediction NLL for one batch."""
    ids = input_ids.to(device)
    mask = attention_mask.to(device).float()

    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask.long()).logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = ids[:, 1:].contiguous()
    shift_mask = mask[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    nll = -torch.gather(log_probs, -1, shift_labels[:, :, None]).squeeze(-1)
    return nll * shift_mask, shift_mask


def eval_ar(args):
    """Evaluate an autoregressive (GPT-2) model."""
    print(f"Loading AR model from {args.model_name_or_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = torch.device(args.device)
    model.eval().to(device)
    print(f"  vocab_size={tokenizer.vocab_size}")

    val_texts = load_val_texts(args.data_dir)
    dl = make_dataloader(val_texts, tokenizer, args.seq_length,
                         args.batch_size)
    metrics = create_metrics(device)

    def compute_fn(batch):
        return compute_causal_nll(
            model, batch["input_ids"], batch["attention_mask"], device)

    results = eval_loop(compute_fn, dl, metrics, desc="AR NLL")
    results["method"] = "causal_nll"
    return results


# ===================================================================
# 3. GPT-BERT evaluation (masked-focus & causal-focus)
# ===================================================================

def compute_masked_ce(model, input_ids, attention_mask, device,
                      mask_token_id, mask_rate=0.15, n_seeds=5):
    """Masked cross-entropy averaged over *n_seeds* random mask patterns."""
    total_nlls = torch.zeros_like(input_ids, dtype=torch.float32,
                                  device=device)
    total_mask = torch.zeros_like(input_ids, dtype=torch.float32,
                                  device=device)
    ids = input_ids.to(device)
    attn = attention_mask.to(device)

    for seed in range(n_seeds):
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(seed + 42)
        rand = torch.rand(ids.shape, device=device)
        torch.random.set_rng_state(rng_state)
        masked_positions = (rand < mask_rate) & attn.bool()

        masked_ids = ids.clone()
        masked_ids[masked_positions] = mask_token_id

        with torch.no_grad():
            logits = model(input_ids=masked_ids,
                           attention_mask=attn).logits

        log_probs = F.log_softmax(logits, dim=-1)
        nll = -torch.gather(log_probs, -1, ids[:, :, None]).squeeze(-1)

        total_nlls += nll * masked_positions.float()
        total_mask += masked_positions.float()

    return total_nlls, total_mask


def compute_pseudo_ll(model, input_ids, attention_mask, device,
                      mask_token_id, spacing=64):
    """Pseudo-log-likelihood with batched masking.

    Positions spaced *spacing* apart are masked simultaneously to keep
    the number of forward passes at *spacing* per batch.  Wider spacing
    reduces cross-position interference at the cost of more passes.
    """
    B, T = input_ids.shape
    ids = input_ids.to(device)
    attn = attention_mask.to(device).float()

    total_nlls = torch.zeros(B, T, device=device)

    for offset in range(min(spacing, T)):
        positions = list(range(offset, T, spacing))
        masked_ids = ids.clone()
        for pos in positions:
            masked_ids[:, pos] = mask_token_id

        with torch.no_grad():
            logits = model(input_ids=masked_ids,
                           attention_mask=attn.long()).logits

        log_probs = F.log_softmax(logits, dim=-1)
        for pos in positions:
            total_nlls[:, pos] = -torch.gather(
                log_probs[:, pos, :], -1,
                ids[:, pos : pos + 1]).squeeze(-1)

    return total_nlls * attn, attn


def eval_gpt_bert(args):
    """Evaluate a GPT-BERT model with up to three NLL methods."""
    print(f"Loading GPT-BERT from {args.model_name_or_path} ...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = torch.device(args.device)
    print(f"  vocab_size={tokenizer.vocab_size}")

    val_texts = load_val_texts(args.data_dir)
    dl = make_dataloader(val_texts, tokenizer, args.seq_length,
                         args.batch_size)

    methods = (args.methods.split(",") if args.methods
               else ["causal", "masked_ce", "pll"])
    all_results = {}

    mask_token_id = tokenizer.mask_token_id

    # --- Causal NLL (uses CausalLM head with upper-triangular mask) ------
    if "causal" in methods:
        print("  [causal] loading CausalLM variant ...")
        causal_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16)
        causal_model.eval().to(device)

        metrics = create_metrics(device)

        def causal_fn(batch):
            return compute_causal_nll(
                causal_model, batch["input_ids"],
                batch["attention_mask"], device)

        res = eval_loop(causal_fn, dl, metrics, desc="Causal NLL")
        res["method"] = "causal_nll"
        all_results["causal"] = res
        del causal_model
        torch.cuda.empty_cache()

    # --- Masked CE -------------------------------------------------------
    if "masked_ce" in methods:
        if mask_token_id is None:
            print("  [masked_ce] SKIP -- tokenizer has no mask token")
        else:
            print("  [masked_ce] loading MaskedLM variant ...")
            masked_model = AutoModelForMaskedLM.from_pretrained(
                args.model_name_or_path, trust_remote_code=True,
                torch_dtype=torch.bfloat16)
            masked_model.eval().to(device)

            metrics = create_metrics(device)

            def mce_fn(batch):
                return compute_masked_ce(
                    masked_model, batch["input_ids"],
                    batch["attention_mask"], device,
                    mask_token_id, mask_rate=0.15, n_seeds=5)

            res = eval_loop(mce_fn, dl, metrics, desc="Masked CE")
            res["method"] = "masked_ce_15pct"
            all_results["masked_ce"] = res
            del masked_model
            torch.cuda.empty_cache()

    # --- Pseudo-log-likelihood -------------------------------------------
    if "pll" in methods:
        if mask_token_id is None:
            print("  [pll] SKIP -- tokenizer has no mask token")
        else:
            print("  [pll] loading MaskedLM variant ...")
            masked_model = AutoModelForMaskedLM.from_pretrained(
                args.model_name_or_path, trust_remote_code=True,
                torch_dtype=torch.bfloat16)
            masked_model.eval().to(device)

            pll_bs = max(1, args.batch_size // 4)
            pll_dl = make_dataloader(val_texts, tokenizer,
                                     args.seq_length, pll_bs)
            metrics = create_metrics(device)

            def pll_fn(batch):
                return compute_pseudo_ll(
                    masked_model, batch["input_ids"],
                    batch["attention_mask"], device,
                    mask_token_id, spacing=args.pll_spacing)

            res = eval_loop(pll_fn, pll_dl, metrics, desc="PLL")
            res["method"] = "pseudo_log_likelihood"
            res["pll_spacing"] = args.pll_spacing
            all_results["pll"] = res
            del masked_model
            torch.cuda.empty_cache()

    return all_results


# ===================================================================
# Inspect mode -- print state-dict keys on the HPC for debugging
# ===================================================================

def inspect_model(args):
    """Download and print state-dict keys for a HuggingFace model."""
    subfolder = args.checkpoint_subfolder or None
    print(f"Inspecting {args.model_name_or_path}"
          f"{f' (subfolder={subfolder})' if subfolder else ''} ...")

    from huggingface_hub import hf_hub_download
    bin_path = hf_hub_download(
        repo_id=args.model_name_or_path,
        filename="pytorch_model.bin",
        subfolder=subfolder)
    sd = torch.load(bin_path, map_location="cpu", weights_only=False)
    print(f"\nKeys ({len(sd)}):")
    for k in sorted(sd.keys()):
        v = sd[k]
        shape = tuple(v.shape) if hasattr(v, "shape") else "n/a"
        print(f"  {k:60s}  {str(shape):>30s}")


# ===================================================================
# CLI
# ===================================================================

def main():
    p = argparse.ArgumentParser(
        description="Evaluate HuggingFace baselines on BabyLM val set")
    p.add_argument("--model_type", required=True,
                   choices=["diffusion", "ar", "gpt_bert", "inspect"],
                   help="Model family to evaluate")
    p.add_argument("--model_name_or_path", required=True,
                   help="HuggingFace repo id or local path")
    p.add_argument("--checkpoint_subfolder", default=None,
                   help="Subfolder inside the HF repo (e.g. chck_100M)")
    p.add_argument("--data_dir", default=None,
                   help="BabyLM data dir with .train/.val files")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seq_length", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_json", default=None)
    p.add_argument("--methods", default=None,
                   help="GPT-BERT methods (comma-sep): causal,masked_ce,pll")
    p.add_argument("--pll_spacing", type=int, default=64,
                   help="PLL: gap between simultaneously-masked positions "
                        "(higher = slower but more accurate)")
    args = p.parse_args()

    if args.model_type == "inspect":
        inspect_model(args)
        return

    if args.data_dir is None:
        p.error("--data_dir is required for evaluation")

    print("=" * 60)
    print(f"  model_type : {args.model_type}")
    print(f"  model      : {args.model_name_or_path}")
    if args.checkpoint_subfolder:
        print(f"  subfolder  : {args.checkpoint_subfolder}")
    print(f"  data_dir   : {args.data_dir}")
    print(f"  batch_size : {args.batch_size}")
    print(f"  seq_length : {args.seq_length}")
    print(f"  device     : {args.device}")
    print("=" * 60)

    t0 = time.time()

    if args.model_type == "diffusion":
        results = eval_diffusion(args)
    elif args.model_type == "ar":
        results = eval_ar(args)
    elif args.model_type == "gpt_bert":
        results = eval_gpt_bert(args)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    elapsed = time.time() - t0

    output = {
        "model_type": args.model_type,
        "model_name_or_path": args.model_name_or_path,
        "checkpoint_subfolder": args.checkpoint_subfolder,
        "results": results,
        "elapsed_seconds": round(elapsed, 1),
        "config": {
            "batch_size": args.batch_size,
            "seq_length": args.seq_length,
        },
    }

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(json.dumps(output, indent=2))

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)),
                    exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()
