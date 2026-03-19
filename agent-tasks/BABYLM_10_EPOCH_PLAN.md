# BabyLM 10-epoch budget – plan

## Constraint

BabyLM rules allow **at most 10 epochs** of training (10 full passes over the data). With a hybrid pipeline (MDLM → EBM), we need to stay within that budget.

## Chosen approach: 10 epochs MDLM, then choose checkpoint for EBM

- **MDLM script:** Train the diffusion model for the **full 10 epochs** (uses the entire BabyLM budget).
- **EBM script:** Does not get additional epochs over the data. You **choose which checkpoint** from the 10-epoch MDLM run to load as the EBM backbone:
  - **Final:** `last.ckpt` (end of 10 epochs)
  - **Earlier:** Any step or epoch checkpoint from the same run (e.g. epoch 5, or a specific step file)

So the only variability is **which checkpoint** the EBM starts from, not how to split epochs between the two stages.

## Implementation

1. **MDLM script (`job_train_babylm_hopper.slurm`)**
   - `trainer.max_epochs=10` (no `max_steps` override; config default is high so epochs are the limit).
   - Comments state that the EBM script will choose a checkpoint from this run.

2. **Energy script (`job_train_babylm_energy_hopper.slurm`)**
   - Requires `BABYLM_CKPT` (path to a diffusion checkpoint). User passes it at submit time.
   - No epoch-split logic; user can point to `last.ckpt` or any other checkpoint from the MDLM run.
   - If challenge rules count EBM finetuning as extra data passes, cap EBM training via `trainer.max_epochs` or `trainer.max_steps` overrides as needed; otherwise the script leaves EBM training length to the user.

3. **Docs**
   - HOPPER_SETUP.md: describe “BabyLM: 10 epochs for MDLM; EBM loads a chosen checkpoint from that run.”

## Summary

| Stage | Script | Role |
|-------|--------|------|
| MDLM | `job_train_babylm_hopper.slurm` | Train for 10 epochs (full budget). Checkpoints saved over the run. |
| EBM | `job_train_babylm_energy_hopper.slurm` | Set `BABYLM_CKPT` to the checkpoint you want (early, mid, or final from the MDLM run). |

No epoch splitting; one 10-epoch MDLM run, then choose which checkpoint the EBM starts from.
