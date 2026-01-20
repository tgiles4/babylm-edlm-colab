# ============================================================================
# Colab Notebook Cell - Copy and paste this entire block into a Colab cell
# ============================================================================

import os
import subprocess
import sys

from IPython.display import HTML, display

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
# MODIFY THIS: Your training command
# ============================================================================
# Examples:
#   "python main.py data=babylm use_energy=False dataset_size=10M noise=cosine"
#   "python main.py data=babylm use_energy=True dataset_size=100M noise=cosine data.data_dir=/content/drive/MyDrive/babylm-edlm/data/train_100M"
#   "python main.py data=openwebtext-split model=small loader.batch_size=16"

training_command = "python main.py data=babylm use_energy=False dataset_size=10M noise=cosine"

# ============================================================================
# Run training (don't modify below)
# ============================================================================
os.chdir('/content/Energy-Diffusion-LLM')
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

