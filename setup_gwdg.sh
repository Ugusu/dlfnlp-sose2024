#!/bin/bash
set -e

# Set up Conda, install Python
module load anaconda3

# Create a new Conda environment using only the default channels
conda create -n dnlp python=3.10 --override-channels -c defaults -y

# Activate the new environment
source activate dnlp

# Install packages using specified channels
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 --override-channels -c defaults -c pytorch -c nvidia -y
conda install tqdm==4.66.2 requests==2.31.0 transformers==4.38.2 tensorboard==2.16.2 tokenizers==0.15.1 --override-channels -c defaults -c conda-forge -c huggingface -y

# Install additional packages using pip
pip install explainaboard-client==0.1.4 sacrebleu==2.4.0

# Download pre-trained models on login node
python -c "from transformers import BertTokenizer, BertModel; BertTokenizer.from_pretrained('bert-base-uncased'); BertModel.from_pretrained('bert-base-uncased')"
python -c "from transformers import AutoTokenizer, BartModel; AutoTokenizer.from_pretrained('facebook/bart-base'); BartModel.from_pretrained('facebook/bart-base')"
