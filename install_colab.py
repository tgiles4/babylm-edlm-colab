"""
Installation script for Google Colab
Run this in a Colab cell with: exec(open('install_colab.py').read())
Or copy-paste the contents into a Colab cell.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a shell command and print status."""
    print(f"\n{'='*60}")
    print(f"Installing: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print(f"✓ Success: {description}")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"✗ Error: {description}")
        print(result.stderr)
        return False
    return True

def main():
    """Main installation function."""
    print("Starting Energy-Diffusion-LLM installation for Colab...")
    print(f"Python version: {sys.version}")

    # Get the codebase directory
    codebase_dir = "/content/Energy-Diffusion-LLM"
    if not os.path.exists(codebase_dir):
        print(f"\n⚠ Warning: Codebase directory not found at {codebase_dir}")
        print("Please make sure you've uploaded and extracted the codebase.")
        print("You can change codebase_dir in this script if needed.")
        return

    os.chdir(codebase_dir)
    print(f"Working directory: {os.getcwd()}")

    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("\n✗ Error: requirements.txt not found!")
        print("Please make sure you're in the correct directory.")
        return

    # Step 1: Install ninja (required build tool)
    success = run_command(
        "pip install -q ninja",
        "ninja (build tool)"
    )
    if not success:
        print("\n✗ Failed to install ninja. This is required for compiling CUDA extensions.")
        return

    # Step 2: Install base requirements
    success = run_command(
        "pip install -q -r requirements.txt",
        "Base requirements"
    )
    if not success:
        print("\n⚠ Warning: Some packages in requirements.txt may have failed.")
        print("Continuing with manual installation of compiled packages...")

    # Step 3: Install flash-attention (REQUIRED - used by dit.py and autoregressive.py)
    print("\n⚠ Note: flash-attention installation may take 5-10 minutes...")
    success = run_command(
        "pip install -q flash-attn --no-build-isolation",
        "flash-attention (REQUIRED, this may take a while)"
    )
    if not success:
        print("\n✗ Error: flash-attention installation failed!")
        print("This is REQUIRED for DIT and autoregressive models.")
        print("You can try: pip install flash-attn --no-build-isolation --no-cache-dir")
        return

    # Step 4: Install causal-conv1d (REQUIRED - used by dimamba.py)
    success = run_command(
        "pip install -q causal-conv1d",
        "causal-conv1d (REQUIRED for DiMamba)"
    )
    if not success:
        print("\n✗ Error: causal-conv1d installation failed!")
        print("This is REQUIRED for DiMamba models.")
        return

    # Step 5: Install mamba-ssm (REQUIRED - used by dimamba.py)
    print("\n⚠ Note: mamba-ssm installation may take a few minutes...")
    success = run_command(
        "pip install -q mamba-ssm",
        "mamba-ssm (REQUIRED for DiMamba, this may take a while)"
    )
    if not success:
        print("\n✗ Error: mamba-ssm installation failed!")
        print("This is REQUIRED for DiMamba models.")
        return

    # Step 4: Verify installation
    print("\n" + "="*60)
    print("Verifying installation...")
    print("="*60)

    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("✗ PyTorch not found")

    try:
        import lightning as L
        print(f"✓ Lightning: {L.__version__}")
    except ImportError:
        print("✗ Lightning not found")

    try:
        import transformers
        print(f"✓ Transformers: {transformers.__version__}")
    except ImportError:
        print("✗ Transformers not found")

    try:
        import datasets
        print(f"✓ Datasets: {datasets.__version__}")
    except ImportError:
        print("✗ Datasets not found")

    try:
        import flash_attn
        print(f"✓ flash-attention: {flash_attn.__version__ if hasattr(flash_attn, '__version__') else 'installed'}")
    except ImportError:
        print("✗ flash-attention: not installed (REQUIRED)")

    try:
        import causal_conv1d
        print(f"✓ causal-conv1d: installed")
    except ImportError:
        print("✗ causal-conv1d: not installed (REQUIRED for DiMamba)")

    try:
        import mamba_ssm
        print(f"✓ mamba-ssm: installed")
    except ImportError:
        print("✗ mamba-ssm: not installed (REQUIRED for DiMamba)")

    print("\n" + "="*60)
    print("Installation complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Mount Google Drive: from google.colab import drive; drive.mount('/content/drive')")
    print("2. Verify tokenizer is in Google Drive")
    print("3. Start training: python main.py data=babylm noise=cosine")

if __name__ == "__main__":
    main()

