#!/bin/bash
# Conda installation script for Google Colab
# Run this in a Colab cell with: !bash install_colab_conda.sh

set -e

echo "Installing Miniconda and setting up environment from requirements.yaml..."

# Step 1: Install Miniconda
echo "Step 1: Installing Miniconda..."
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p /usr/local/miniconda
rm miniconda.sh

# Add conda to PATH
export PATH="/usr/local/miniconda/bin:$PATH"

# Initialize conda
/usr/local/miniconda/bin/conda init bash
source ~/.bashrc

# Step 2: Create environment from requirements.yaml
echo ""
echo "Step 2: Creating conda environment from requirements.yaml..."
cd /content/Energy-Diffusion-LLM

# Note: We'll modify the approach since Colab already has PyTorch/CUDA
# Option A: Create environment but don't install PyTorch (use Colab's)
conda env create -f requirements.yaml -n edlm || {
    echo "Warning: Full environment creation failed. Trying alternative approach..."

    # Option B: Create environment with Python only, then install packages
    conda create -n edlm python=3.9 -y
    conda activate edlm

    # Install conda packages (excluding PyTorch which Colab already has)
    conda install -y -c pytorch -c nvidia -c anaconda \
        jupyter=1.0.0 \
        pip=23.3.1 \
        -n edlm

    # Install pip packages
    conda run -n edlm pip install -r requirements.txt
}

echo ""
echo "Conda environment 'edlm' created!"
echo ""
echo "To activate the environment in Colab, use:"
echo "  import sys"
echo "  sys.path.insert(0, '/usr/local/miniconda/envs/edlm/lib/python3.9/site-packages')"
echo ""
echo "Or restart the runtime and activate:"
echo "  !conda activate edlm"






