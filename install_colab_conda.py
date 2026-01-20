"""
Conda installation script for Google Colab
Run this in a Colab cell with: exec(open('install_colab_conda.py').read())
"""

import os
import subprocess
import sys


def run_command(cmd, description, check=True):
    """Run a shell command and print status."""
    print(f"\n{'='*60}")
    print(f"{description}")
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
        return True
    else:
        print(f"✗ Error: {description}")
        print(result.stderr)
        if check:
            return False
        return True

def main():
    """Main installation function."""
    print("Starting Conda installation for Energy-Diffusion-LLM in Colab...")
    print(f"Python version: {sys.version}")

    # Check if conda is already installed
    conda_installed = subprocess.run(
        "which conda", shell=True, capture_output=True
    ).returncode == 0

    if not conda_installed:
        print("\nStep 1: Installing Miniconda...")
        # Install Miniconda
        run_command(
            "wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh",
            "Downloading Miniconda"
        )
        run_command(
            "bash /tmp/miniconda.sh -b -p /usr/local/miniconda",
            "Installing Miniconda"
        )
        run_command(
            "rm /tmp/miniconda.sh",
            "Cleaning up installer"
        )

        # Add to PATH
        os.environ['PATH'] = '/usr/local/miniconda/bin:' + os.environ.get('PATH', '')
        print("✓ Miniconda installed")
    else:
        print("✓ Conda already installed")
        os.environ['PATH'] = '/usr/local/miniconda/bin:' + os.environ.get('PATH', '')

    # Get codebase directory
    codebase_dir = "/content/Energy-Diffusion-LLM"
    if not os.path.exists(codebase_dir):
        print(f"\n⚠ Warning: Codebase directory not found at {codebase_dir}")
        return

    os.chdir(codebase_dir)
    print(f"\nWorking directory: {os.getcwd()}")

    # Check if requirements.yaml exists
    if not os.path.exists("requirements.yaml"):
        print("\n✗ Error: requirements.yaml not found!")
        return

    print("\nStep 2: Creating conda environment from requirements.yaml...")
    print("⚠ Note: Colab already has PyTorch/CUDA installed.")
    print("We'll create the environment but may skip some packages to avoid conflicts.")

    # Try to create environment
    # Note: We modify the approach because Colab already has PyTorch
    success = run_command(
        "conda env create -f requirements.yaml -n edlm",
        "Creating conda environment",
        check=False
    )

    if not success:
        print("\n⚠ Full environment creation failed. Trying alternative approach...")
        print("Creating environment with Python only, then installing packages separately...")

        # Create environment with just Python
        run_command(
            "conda create -n edlm python=3.9 -y",
            "Creating base environment"
        )

        # Install conda packages (excluding PyTorch)
        run_command(
            "conda install -y -c anaconda jupyter=1.0.0 pip=23.3.1 -n edlm",
            "Installing conda packages"
        )

        # Install pip packages using conda run
        print("\nInstalling pip packages in conda environment...")
        run_command(
            "conda run -n edlm pip install -q ninja",
            "Installing ninja"
        )
        run_command(
            "conda run -n edlm pip install -q -r requirements.txt",
            "Installing pip requirements"
        )
        run_command(
            "conda run -n edlm pip install -q flash-attn --no-build-isolation",
            "Installing flash-attn"
        )
        run_command(
            "conda run -n edlm pip install -q causal-conv1d mamba-ssm",
            "Installing causal-conv1d and mamba-ssm"
        )

    print("\n" + "="*60)
    print("Conda environment 'edlm' created!")
    print("="*60)
    print("\nTo use the environment in Colab:")
    print("1. Add conda environment to Python path:")
    print("   import sys")
    print("   sys.path.insert(0, '/usr/local/miniconda/envs/edlm/lib/python3.9/site-packages')")
    print("\n2. Or use conda run for commands:")
    print("   !conda run -n edlm python main.py data=babylm noise=cosine")
    print("\n3. Or restart runtime and activate:")
    print("   !source /usr/local/miniconda/bin/activate edlm")

if __name__ == "__main__":
    main()






