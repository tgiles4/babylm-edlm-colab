#!/bin/bash
# Installation script for Google Colab
# Run this in a Colab cell with: !bash install_colab.sh

set -e

echo "Installing Energy-Diffusion-LLM dependencies for Colab..."

# Step 1: Install ninja (required build tool for CUDA extensions)
echo "Installing ninja (build tool)..."
pip install -q ninja

# Step 2: Install base requirements (excluding compiled packages)
echo "Installing base requirements..."
pip install -q -r requirements.txt || {
    echo "Warning: Some packages in requirements.txt may have failed."
    echo "Continuing with manual installation of compiled packages..."
}

# Step 2.5: Install triton (REQUIRED for torch.library.wrap_triton used by flash-attn)
echo "Installing triton (required for flash-attn)..."
pip install -q triton>=2.1.0 || {
    echo "Warning: triton installation failed, but continuing..."
}

# Step 3: Install flash-attention (REQUIRED - used by dit.py and autoregressive.py)
# This requires CUDA and ninja, may take 5-10 minutes
echo "Installing flash-attention (REQUIRED, this may take 5-10 minutes)..."
pip install -q flash-attn --no-build-isolation || {
    echo "Error: flash-attention installation failed!"
    echo "This is REQUIRED for DIT and autoregressive models."
    exit 1
}

# Step 4: Install causal-conv1d (REQUIRED - used by dimamba.py)
echo "Installing causal-conv1d (REQUIRED for DiMamba models)..."
pip install -q causal-conv1d || {
    echo "Error: causal-conv1d installation failed!"
    echo "This is REQUIRED for DiMamba models."
    exit 1
}

# Step 5: Install mamba-ssm (REQUIRED - used by dimamba.py)
echo "Installing mamba-ssm (REQUIRED for DiMamba models, this may take a few minutes)..."
pip install -q mamba-ssm || {
    echo "Error: mamba-ssm installation failed!"
    echo "This is REQUIRED for DiMamba models."
    exit 1
}

# Verify PyTorch installation
echo ""
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Verify required imports
echo ""
echo "Verifying required model dependencies..."
python -c "
try:
    import flash_attn
    print('✓ flash-attn imported successfully')
except ImportError as e:
    print('✗ flash-attn import failed:', e)
    exit(1)

try:
    import causal_conv1d
    print('✓ causal-conv1d imported successfully')
except ImportError as e:
    print('✗ causal-conv1d import failed:', e)
    exit(1)

try:
    import mamba_ssm
    print('✓ mamba-ssm imported successfully')
except ImportError as e:
    print('✗ mamba-ssm import failed:', e)
    exit(1)
"

echo ""
echo "Installation complete!"

