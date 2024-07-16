Here's a structured README template for your group project:

---

# G10 Neural Wordsmiths - DNLP SS24 Final Project

<div align="left">
<b>Group Name:</b> <b style="color:yellow;"> Neural Wordsmiths </b><br/><br/>
<b>Group Code:</b> G10<br/><br/>
<b>Group Repository:</b> <a href="https://github.com/Ugusu/dlfnlp-sose2024">Ugusu/dlfnlp-sose2024</a><br/><br/>
<b>Tutor Responsible:</b> Finn<br/><br/>
<b>Group Team Leader:</b> Ughur Mammadzada<br/><br/>
<b>Group Members:</b> Amirreza Aleyasin, Daniel Ariza, Pablo Jahnen, Enno Weber
</div>


Sure, here's the revised introduction with the BART information included and the second part of the project described in the future tense:

---

## Introduction

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.2.0-orange.svg)](https://pytorch.org/)
[![Apache License 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)
[![Code Style](https://img.shields.io/badge/Code%20Style-PEP8-green.svg)](https://www.python.org/dev/peps/pep-0008/)
[![AI-Usage Card](https://img.shields.io/badge/AI_Usage_Card-pdf-blue.svg)](./AI-Usage-Card.pdf/)

This repository contains the official implementation of our final project for the Deep Learning for Natural Language Processing course at the University of Göttingen. The project involved implementing components of the BERT model and applying it to tasks like sentiment classification, paraphrase detection, and semantic similarity. Additionally, we implemented a BART model for paraphrase type generation and detection.

The project is divided into two main parts:

- **Part 01:** We implemented key aspects of the BERT model, including multi-head self-attention and Transformer layers, and applied it to tasks such as sentiment analysis, question similarity, and semantic similarity. Additionally, we implemented a BART model for paraphrase type generation and detection. Each task had at least one baseline implementation.
- **Part 02:** We will fine-tune and extend the models to improve performance on the same downstream tasks. Several techniques from recent research papers will be explored to create more robust and semantically-rich sentence embeddings, aiming to improve over the baseline implementations.

The initial part focused on establishing a working baseline for each task, while the latter part will concentrate on refining and optimizing these models for better performance.

---

## Setup Instructions

To set up the environment and install dependencies for local development and testing, use the provided bash script `setup.sh`. This script creates a new conda environment called `dnlp` and installs all required packages. It will also check for CUDA and install the appropriate PyTorch version accordingly.

```sh
setup.sh
```

Activate the environment with:

```sh
conda activate dnlp
```

For setting up the repository on the remote GWDG cluster for training models with GPUs via SSH connection, use the `setup_gwdg.sh` script. This script is specifically designed to configure the environment for GPU-accelerated training on the GWDG cluster.

```sh
setup_gwdg.sh
```

---

## Training

To train the model, activate the environment and run:

```sh
python -u multitask_classifier.py --use_gpu
```

Important parameters and their descriptions can be seen by running:

```sh
python multitask_classifier.py --help
```

## Evaluation

The model is evaluated after each epoch on the validation set. Results are printed to the console and saved in the `logdir` directory. The best model is saved in the `models` directory.

## Methodology

In this section, describe the process and methods used in the project. Briefly explain the ideas implemented to improve the model. Make sure to indicate how existing ideas were used and extended.

## Experiments

Detail the experiments conducted, including tasks and models considered. Describe each experiment with the following points:

- What experiments are being executed?
- What were your expectations?
- What changes were made compared to the base model?
- What were the results?
- Add relevant metrics and plots.
- Discuss the results.

## Results

Summarize the results of your experiments in tables:

| **Task** | **Metric 1** | **Metric n** |
|----------|---------------|--------------|
| Baseline | 45.23%        | ...          |
| Improvement 1 | 58.56%   | ...          |
| Improvement 2 | 52.11%   | ...          |
| ...      | ...           | ...          |

Discuss your results, observations, correlations, etc.

## Hyperparameter Optimization

Briefly describe how you optimized your hyperparameters. If you focused strongly on hyperparameter optimization, include it in the Experiment section.

## Visualizations

Add relevant graphs showing metrics like accuracy, validation loss, etc., during training. Compare different training processes of your improvements in these graphs.

## Members Contribution

Explain the contribution of each group member:

**Daniel Ariza:** ...

**Amirreza Aleyasin:** ...

**Pablo Jahnen:** ...

**Ughur Mammadzada:** ...

**Enno Weber:** ...

## AI-Usage Card

Artificial Intelligence (AI) aided the development of this project. For transparency, we provide our [AI-Usage Card](./AI-Usage-Card.pdf/) at the top.

## References

List all references (repositories, papers, etc.) used for your project.

## Acknowledgement

The project description, partial implementation, and scripts were adapted from the final project for the Stanford [CS 224N class](https://web.stanford.edu/class/cs224n/). The BERT implementation was adapted from the "minbert" assignment at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html). Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library. Scripts and code were modified by [Jan Philip Wahle](https://jpwahle.com/) and [Terry Ruas](https://terryruas.com/).

