# ============================================================================
# Colab Notebook Cell - Copy and paste this entire block into a Colab cell
# Run from repo root so __file__ resolves; or set REPO_ROOT below if using exec()
# ============================================================================

import os
import subprocess
import sys

from IPython.display import HTML, display

# Repo root: parent of colab/ (when script path is available)
try:
    _script_path = os.path.abspath(__file__)
    REPO_ROOT = os.path.dirname(os.path.dirname(_script_path))
except NameError:
    REPO_ROOT = "/content/babylm-edlm-colab"  # fallback when exec() from Colab

# JavaScript to keep Colab connection alive (clicks "Connect" button every 60 seconds)
keep_alive_js = """
<script>
function ClickConnect(){
    console.log("Keeping Colab connection alive...");
    document.querySelector("colab-toolbar-button#connect").click();
}
setInterval(ClickConnect, 60000);
</script>
"""
display(HTML(keep_alive_js))

# ============================================================================
# MODIFY THIS: Your training command (use Colab overrides from colab/COLAB.md)
# ============================================================================
# Examples with Colab/Drive paths:
#   "python main.py data=babylm use_energy=False dataset_size=10M noise=cosine data.data_dir=/content/drive/MyDrive/babylm-edlm/data/train_10M data.tokenizer_name_or_path=/content/drive/MyDrive/babylm-edlm/tokenizer/tokenizer.json data.cache_dir=/content/drive/MyDrive/babylm-edlm/cache checkpointing.save_dir=/content/drive/MyDrive/babylm-edlm/checkpoints hydra.run.dir=/content/drive/MyDrive/babylm-edlm/outputs wandb.save_dir=/content/drive/MyDrive/babylm-edlm/wandb"
#   "python main.py data=babylm use_energy=True dataset_size=100M noise=cosine data.data_dir=/content/drive/MyDrive/babylm-edlm/data/train_100M ..." (same overrides)

training_command = "python main.py data=babylm use_energy=False dataset_size=10M noise=cosine"

# ============================================================================
# Run training (don't modify below)
# ============================================================================
os.chdir(REPO_ROOT)
print(f"Starting: {training_command}")
print("Connection kept alive automatically (clicks every 60s)")
print("=" * 80)

process = subprocess.Popen(
    training_command.split(),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True,
    bufsize=1
)

for line in process.stdout:
    print(line, end='')
    sys.stdout.flush()

process.wait()
print(f"\n{'=' * 80}")
print(f"Training finished with exit code: {process.returncode}")
