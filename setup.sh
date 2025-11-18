#!/bin/bash

# Setup script for Gemini Robotics Pointing Workshop

ENV_NAME="gemini-robotics-pointing"
PYTHON_VERSION="3.10"

echo "🤖 Setting up environment: $ENV_NAME"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda could not be found. Please install Miniforge or Anaconda first."
    echo "   See: https://github.com/conda-forge/miniforge?tab=readme-ov-file#install"
    exit 1
fi

# Create Conda environment
echo "📦 Creating Conda environment..."
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

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
echo "   You can now configure 'config.py' and run the workshop script:"
echo "   python workshop.py"
