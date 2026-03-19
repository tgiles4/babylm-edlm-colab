# Google Colab setup (optional)

This folder contains Colab-only install and run scripts. The main codebase defaults to Hopper/cluster paths; use the overrides below when running in Colab.

## Running the install script

From the **repo root** (e.g. after cloning or extracting to `/content/babylm-edlm-colab`):

**Pip install (recommended):**
```python
# In a Colab cell:
!cd /content/babylm-edlm-colab && bash colab/install_colab.sh
```
Or using the Python installer:
```python
exec(open('/content/babylm-edlm-colab/colab/install_colab.py').read())
```

**Conda install (advanced):**
```python
!cd /content/babylm-edlm-colab && bash colab/install_colab_conda.sh
# or
exec(open('/content/babylm-edlm-colab/colab/install_colab_conda.py').read())
```

Mount Google Drive first if you store data/tokenizer there:
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Required Hydra overrides for Colab

Default config uses `$SCRATCH` (or `./scratch`). In Colab you must override paths so data, tokenizer, cache, checkpoints, and outputs point to your Colab/Drive locations.

Use a **base directory** on Drive, e.g.:
- `/content/drive/MyDrive/babylm-edlm`

Then pass these overrides when calling `main.py`:

| Override | Example (copy-paste) |
|----------|----------------------|
| `data.data_dir` | `data.data_dir=/content/drive/MyDrive/babylm-edlm/data/train_10M` |
| `data.tokenizer_name_or_path` | `data.tokenizer_name_or_path=/content/drive/MyDrive/babylm-edlm/tokenizer/tokenizer.json` |
| `data.cache_dir` | `data.cache_dir=/content/drive/MyDrive/babylm-edlm/cache` |
| `checkpointing.save_dir` | `checkpointing.save_dir=/content/drive/MyDrive/babylm-edlm/checkpoints` |
| `hydra.run.dir` | `hydra.run.dir=/content/drive/MyDrive/babylm-edlm/outputs/${data.train}/${dataset_size}/${use_energy}` |
| `wandb.save_dir` | `wandb.save_dir=/content/drive/MyDrive/babylm-edlm/wandb` |

**Full example (10M, standard DIT):**
```bash
python main.py data=babylm use_energy=False dataset_size=10M noise=cosine \
  data.data_dir=/content/drive/MyDrive/babylm-edlm/data/train_10M \
  data.tokenizer_name_or_path=/content/drive/MyDrive/babylm-edlm/tokenizer/tokenizer.json \
  data.cache_dir=/content/drive/MyDrive/babylm-edlm/cache \
  checkpointing.save_dir=/content/drive/MyDrive/babylm-edlm/checkpoints \
  hydra.run.dir=/content/drive/MyDrive/babylm-edlm/outputs/babylm/10M/False \
  wandb.save_dir=/content/drive/MyDrive/babylm-edlm/wandb
```

**For 100M data**, change the data dir:
```bash
data.data_dir=/content/drive/MyDrive/babylm-edlm/data/train_100M
```

Ensure on Drive you have:
- `tokenizer/tokenizer.json` (from `train_tokenizer.py` or your tokenizer)
- `data/train_10M/` (and optionally `data/train_100M/`) with BabyLM `.train` files

## Long-running training in Colab

Use `colab/colab_train_cell.py`: copy its contents into a notebook cell. It keeps the connection alive and runs your training command. Edit the `training_command` variable to include the overrides above.
