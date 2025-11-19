#!/bin/bash
set -e

# Configuration
ENV_NAME="gemini-robotics-pointing"
PYTHON_VERSION="3.10"

# Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Miniforge or Anaconda first."
    echo "   https://github.com/conda-forge/miniforge?tab=readme-ov-file#install"
    exit 1
fi

# Create Conda environment if it doesn't exist
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "✅ Conda environment '$ENV_NAME' already exists."
else
    echo "📦 Creating Conda environment..."
    conda create -y -n $ENV_NAME python=$PYTHON_VERSION
fi

# Activate environment (this trick allows activating in a script)
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install ffmpeg
echo "🎥 Installing ffmpeg..."
conda install -y ffmpeg -c conda-forge

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install -r requirements.txt

echo "✅ Setup complete!"
echo "You can now configure 'config.py' and run the workshop script:"
echo "   conda activate $ENV_NAME"
echo "   python workshop.py"
