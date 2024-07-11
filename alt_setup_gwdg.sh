#!/bin/bash
set -e

# Load the necessary Python module
module load python/3.10.13

# Remove the existing virtual environment if it exists
if [ -d "$HOME/dnlp_venv" ]; then
    rm -rf ~/dnlp_venv
fi

# Create a virtual environment named 'dnlp_venv'
python -m venv ~/dnlp_venv

# Activate the virtual environment
source ~/dnlp_venv/bin/activate

# Upgrade pip to the latest version
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --extra-index-url https://download.pytorch.org/whl/cu121

# Install additional packages
pip install tqdm==4.66.2 requests==2.31.0 transformers==4.38.2 tensorboard tokenizers==0.15.1
pip install explainaboard-client==0.1.4 sacrebleu==2.4.0

# Download pre-trained models on login node
python -c "from transformers import BertTokenizer, BertModel; BertTokenizer.from_pretrained('bert-base-uncased'); BertModel.from_pretrained('bert-base-uncased')"
python -c "from transformers import AutoTokenizer, BartModel; AutoTokenizer.from_pretrained('facebook/bart-base'); BartModel.from_pretrained('facebook/bart-base')"
