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

---

## Introduction

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.2.0-orange.svg)](https://pytorch.org/)
[![Apache License 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)
[![Code Style](https://img.shields.io/badge/Code%20Style-PEP8-green.svg)](https://www.python.org/dev/peps/pep-0008/)
[![AI-Usage Card](https://img.shields.io/badge/AI_Usage_Card-pdf-blue.svg)](./NeuralWordsmiths_AI_Usage_Card.pdf/)

This repository contains the official implementation of our final project for the Deep Learning for Natural Language Processing course at the University of Göttingen. The project involved implementing components of the BERT model and applying it to tasks like sentiment classification, paraphrase detection, and semantic similarity. Additionally, we implemented a BART model for paraphrase type generation and detection.

The project is divided into two main parts:

- **Part 01:** We implemented key aspects of the BERT model, including multi-head self-attention and Transformer layers, and applied it to tasks such as sentiment analysis, question similarity, and semantic similarity. Additionally, we implemented a BART model for paraphrase type generation and detection. Each task had at least one baseline implementation.
- **Part 02 (in progress):** We will fine-tune and extend the models to improve performance on the same downstream tasks. Several techniques from recent research papers will be explored to create more robust and semantically-rich sentence embeddings, aiming to improve over the baseline implementations.

The initial part focused on establishing a working baseline for each task, while the latter part will concentrate on refining and optimizing these models for better performance.

---

## Setup Instructions

To set up the environment and install dependencies for local development and testing, use the provided bash script `setup.sh`. This script creates a new conda environment called `dnlp` and installs all required packages. It will also check for CUDA and install the appropriate PyTorch version accordingly.

```sh
bash setup.sh
```

Activate the environment with:

```sh
conda activate dnlp
```

For setting up the repository on the remote GWDG cluster for training models with GPUs via SSH connection, use the `setup_gwdg.sh` script. This script is specifically designed to configure the environment for GPU-accelerated training on the GWDG cluster.

```sh
bash setup_gwdg.sh
```

---

## Training
### Local: 

To train the model, activate the environment and run:

```sh
python -u multitask_classifier.py --use_gpu
```

There are a lot of parameters that can be set. The most important ones are:

| Parameter               | Description                                                                           |
|-------------------------|---------------------------------------------------------------------------------------|
| `--task`                | Choose between `"sst"`, `"sts"`, `"qqp"`, `"multitask"` to train for different tasks. |
| `--seed`                | Random seed for reproducibility.                                                      |
| `--epochs`              | Number of epochs to train the model.                                                  |
| `--option`              | Determines if BERT parameters are frozen (`pretrain`) or updated (`finetune`).        |
| `--use_gpu`             | Whether to use the GPU for training.                                                  |
| `--subset_size`         | Number of examples to load from each dataset for testing.                             |
| `--context_layer`       | Include context layer if this flag is set.                                            |
| `--regularize_context`  | Use regularized context layer variant if this flag is set.                            |
| `--pooling`             | Choose the pooling strategy: `"cls"`, `"average"`, `"max"`, or `"attention"`.         |
| `--optimizer`           | Optimizer to use.                                                                     |
| `--batch_size`          | Batch size for training, recommended 64 for 12GB GPU.                                 |
| `--hidden_dropout_prob` | Dropout probability for hidden layers.                                                |
| `--lr`                  | Learning rate, defaults to `1e-3` for `pretrain`, `1e-5` for `finetune`.              |
| `--local_files_only`    | Force the use of local files only (do not download from remote repositories).         |

All parameters and their descriptions can be seen by running:

```sh
python multitask_classifier.py --help
```

### HPC:
to submit the job to a node in the GWDG HPC cluster, run:
settings can be configured according to the requirements in the `run_train.sh` file.
```sh
sbatch run_train.sh
```
---

## Evaluation

The model is evaluated after each epoch on the validation set. Results are printed to the console and saved in the `logdir` directory. The best model is saved in the `models` directory.

---

## Phase I

We implemented the base BERT and BART for the first phase of the project.

### BERT

For the BERT model we implemented 3 tasks:
- Sentiment Classification: The model got an additional classification layer, which takes as input the embedings from the BERT model. The used dataset is Stanford Sentiment Treebank. Loss function - Cross Entropy.
- Semantic Textual Similarity: Similar to the previous task, a classifier layer was added to the model. It takes as input the model's embedings, and predicts single logit, which defines the similarity score between senteces, which then is normilized to the range 0-5, 5 being most similar and 0 being related. Loss fucntion - Mean Squared Error Loss.
- Paraphrase Detection: The classifier layer at the end is similar to the previous task, with inputs being the embeddings of the model, and output a logit. The logit is normilized to the range 0-1, 1 being "is a paraphrase" and 0 being "not a paraphrase". Loss function - Binary Cross Entropy with Logits.

All embeddings go through a dropout layer, before being passed to the classifier layers.

For multitask training all tasks were run for 10 epochs with AdamW optimizer and hyperparameters:
- Learning rate: 1e-5
- Dropout probability: 0.2
- Batch size: 64
- Epsilon: 1e-8
- Betas: (0.9, 0.999)

For separate fine-tuning per tasks the hyperparameters were the same, except for Paraphrase Detection task, as 1 epoch is enough.

The model was trained on fine-tuning mode, so all parameters were updated.

BERT version: BERT Base Uncased.

### BART

BART has 2 tasks:

- BART_generation: for this task, we used the BART model to generate paraphrases of a given sentence.
  - We used the `BartForConditionalGeneration` model from the `transformers` library the pretrained `bart-large` has been used.
  - The model was trained on the `etpc-paraphrase-train.csv` dataset, which contains 2019 paraphrase pairs.
  - The model was fine-tuned on the `etpc-paraphrase-train.csv` dataset for 5 epochs with a batch size of 16.
  - The model was evaluated on the `etpc-paraphrase-dev.csv` and `etpc-paraphrase-generation-test-student` datasets.

- BART_detection: We used BART-large model to detect 7 differenct paraphrase types given a sentence. Tokenization was done using AutoTokenizer from the transformers library and using a pretrained BartModel from the same library, the model was fine-tuned on the etpc-paraphrase-train.csv using AdamW optimzer and CrossEntropyLoss loss function and validated on the etpc-paraphrase-dev.csv dataset for 5 epochs, learning rate 1e-5 and batch size 16. It is saved for best validation loss performance and was then tested on the etpc-paraphrase-generation-test-student dataset.

BART version: BART Large.

## Experiments

### Learning all tasks vs. Learning one task:

- A BERT model was trained to be able to solve all 3 tasks, and was compared to a BERT model trained on the tasks independetly.
- The results for Sentiment Classification and Semantic Textual Similarity degrade, while for Paraphrase Detection increase.
- Most probable explanation: Dataset sizes are not equal. Later or bigger trainings degrade previous or smaller trainings.
- Possible solution: Trainings on bigger datasets first. Number of epochs relative to dataset size.

---

# Phase II

## Improvements upon Base Models

## 1. Proposals

### 1.1 Contextual Global Attention (CGA)
As part of the improvements for the sentiment analysis task of the project, Contextual Global Attention 
(CGA) was introduced to enhance BERT's performance on the sentiment analysis task and in the pooling of the 
encoded output embeddings. This alternate mechanism aims to enhance BERT's ability to capture long-range 
dependencies by  integrating global context into its self-attention mechanism.

### 1.2 Pooling Strategies
While the **CLS** token is traditionally used as the aggregate representation in BERT's output, it may 
not fully capture the semantic content of an entire sequence. To address this, the following pooling 
strategies were tested:

* CLS token pooling
* Average pooling
* Max pooling
* Attention-based pooling

This experimentation aimed to identify whether these approaches could provide a more comprehensive 
representation, thereby improving the model's performance.

## 2. Methodology

### BERT

### **2.1 Contextual Global Attention (CGA) Layer**

This layer computes a global context vector by averaging token embeddings and refining it through a feed-forward network. The refined context vector is incorporated into BERT’s self-attention mechanism via a custom **Contextual Global Attention (CGA)** mechanism, implemented in the `context_bert.py` file. 

The CGA mechanism introduces additional weight matrices and gating parameters that modulate the influence of the global context on token-level representations. A regularized variant of CGA, implemented as the `GlobalContextLayerRegularized` class, incorporates layer normalization and dropout to enhance model generalization and stability during training.

$$
\begin{bmatrix}
\hat{\mathbf{Q}} \\
\hat{\mathbf{K}}
\end{bmatrix} = (1 - \begin{bmatrix} \lambda_Q \\ \lambda_K \end{bmatrix}) \begin{bmatrix} \mathbf{Q} \\ \mathbf{K} \end{bmatrix} + \begin{bmatrix} \lambda_Q \\ \lambda_K \end{bmatrix} \mathbf{C} \begin{bmatrix} \mathbf{U}_Q \\ \mathbf{U}_K \end{bmatrix}
$$

It was decided to use the average of the hidden states across the entire input sequence as choice of context representation. This is called **"global context"**:

$$
\mathbf{c} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{h}_i
$$

For further details on the workings of the CGA, refer to the original paper and the implementation in the code.

### **2.2 Pooling Strategies**

Different pooling strategies were explored to determine the most effective method for summarizing the information captured by BERT. These strategies included:

1. **CLS Token Pooling:**
The final hidden state of the CLS token, $\mathbf{h}_{\text{CLS}}$, is used as the aggregate representation:

$$
\mathbf{p} = \mathbf{h}_{\text{CLS}}
$$

2. **Average Pooling:**
The hidden states of all tokens are averaged to produce the sentence representation:

$$
\mathbf{p} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{h}_i
$$

where $n$ is the number of tokens in the sequence.

3. **Max Pooling:**
The maximum value across all token hidden states is selected for each dimension:

$$
\mathbf{p}_j = \max_{i} \mathbf{h}_{ij}
$$

where $\mathbf{h}_{ij}$ is the hidden state of token $i$ in dimension $j$.

4. **Attention-Based Pooling:**
Attention scores from the Global Context Layer are used to compute a weighted sum of the hidden states:

$$
\mathbf{p} = \sum_{i=1}^{n} \alpha_i \mathbf{h}_i
$$

where $\(\alpha_i\)$ represents the attention weight assigned to each hidden state.

These pooling strategies were implemented and evaluated to identify the most effective approach for improving performance in sentiment analysis$

---

## **3. Experiments**

### **3.1 Grid Search for Hyperparameter Optimization**

A comprehensive grid search was conducted to identify the optimal hyperparameters for the sentiment 
analysis task, particularly focusing on the integration of the newly introduced Contextual Global Attention. 
The search encompassed various combinations of pooling strategies, learning rates, dropout probabilities, 
batch sizes, epochs, and optimizers, resulting in 192 unique configurations. When combined with the four 
variations in the Contextual Global Attention (see bash scripts below), this amounts to a total of 
768 different configurations.

**Grid Search Configuration:**

- **Pooling Strategies:** `CLS Token`, `Average`, `Max`, `Attention`
- **Learning Rates:** `1e-5`, `5e-5`
- **Hidden Dropout Probabilities:** `0.3`, `0.5`
- **Batch Sizes:** `16`, `32`, `64`
- **Epochs:** `5`, `10`
- **Optimizers:** `AdamW`, `SophiaG`

**Execution Overview:**

The grid search was executed on the HPC cluster (as described in the setup section above), with the following key points:

- **Bash Scripts:** To initiate the grid search for different configurations of the CGA Layer, 
use the provided [bash scripts](sst_grid_search_experiments/experiment_scripts). These scripts correspond to various setups, including options with or 
without the extra layer and with or without regularization.

- **Result Storage:** Results for each grid search run were [saved](sst_grid_search_experiments/sst_experiments_grid_search_results) in JSON format, for analysis of the performance metrics for each configuration.

- **Manual Tuning:** Configuration options within the grid search [Python script](grid_search.py) can be manually adjusted to tweak the parameters being tested (lines 160-165).

- **Submission Script:** An executable script was also provided to manage the submission of all grid search jobs on the HPC cluster at once.

  ```sh
  submit_grid_search_jobs.sh
  ```

We recommend running these scripts from the project's home directory.

---

## **4. Results**

### 4.1 Grid Search, CGA and Attention-based Pooling

#### **4.1.1 Data Overview**

A total of 768 experiments were conducted, all of which successfully completed. The experiments tested various configurations, including pooling strategies, learning rates, dropout probabilities, batch sizes, epochs, and optimizers.

#### **4.1.2 Overall Best SST Accuracy Performance**

The highest SST accuracy achieved was **0.530233** with the following configuration:
- **Pooling Strategy:** `CLS`
- **Extra Context Layer:** `False`
- **Regularize Context:** `True`
- **Learning Rate:** `1e-5`
- **Hidden Dropout Probability:** `0.3`
- **Batch Size:** `64`
- **Optimizer:** `AdamW`
- **Epochs:** `5`

This configuration can be replicated by running the following script:

```sh
best_sst_performance.sh
```

However, due to random variation in training and evaluation, the exact accuracy value may not be equal, but within
the same order of magnitude.

#### **4.1.3 Overall Effect of CGA Layer on SST Performance**

The Global Context Layer showed the following impact on SST accuracy:

| CGA Layer | SST Accuracy (Mean) | SST Accuracy (Max) |
|-----------|---------------------|--------------------|
| False     | 0.430               | 0.530              |
| True      | 0.331               | 0.523              |
| Baseline  | N/A                 | 0.522              |

The higher accuracy of the model without a CGA layer with respect to the baseline lies in the alternate hyperparameter
selection optimized through the grid search.

| **Stanford Sentiment Treebank (SST)**   | **Dev accuracy** |
|-----------------------------------------|------------------|
| Baseline                                | 0.522            | 
| Contextual Global Attention (CGA)       | 0.523            |
| CGA-based Attention-pooling             | 0.522            |
| Using Grid Search Best Results (no CGA) | 0.530            |

The generated [violin plot](sst_grid_search_experiments/analyses_visualizations/impact_cga_sst_accuracy.png) shows that the model without the CGA Layer generally outperformed the one with it, with most
results being concentrated on the ~0.500 mark vs. ~0.300 for the models with the extra CGA layer. However, under certain hyperparameter
selection, accuracy in the same orders of magnitude can be reached.

#### **4.1.4 Effect of CGA Layers and Attention Pooling on SST Performance**

A deeper insight into the effects of regularized and non-regularized CGA layers on SST performance across all experiments
reveals:
- Regularization increases STT accuracy when extra CGA layer is present.
- Attention-based pooling using a CGA layer doesn't improve SST accuracy on average, even when regularized.

Additionally, the best SST performance under different conditions was as follows:

- **With Extra Context Layer:** 0.523 (CLS, Regularize Context: True, AdamW)
- **With Attention Pooling:** 0.522 (Attention, Regularize Context: True, AdamW)
- **With Both:** 0.505 (Regularize Context: True, AdamW)

![alt text](sst_grid_search_experiments/analyses_visualizations/sst_performance_comparison.png)

#### **4.1.4 Effectiveness of Pooling Strategies**


Pooling strategies were evaluated, with the best SST accuracy results being achieved with the standard CLS-token-based 
pooling strategy:

| Pooling Strategy | SST Accuracy (Mean) | SST Accuracy (Max) |
|------------------|---------------------|--------------------|
| CLS (default)    | 0.429               | 0.530              |
| Attention        | 0.427               | 0.522              |
| Average          | 0.424               | 0.512              |
| Max              | 0.421               | 0.510              |

For an illustrative comparison, refer to the corresponding [box plot](sst_grid_search_experiments/analyses_visualizations/sst_accuracy_by_pooling_strategy.png).

#### **Metrics worth exploring further:**
- Model stability
- Training time

#### **Further Details:**
For a more in-depth analysis and additional results, refer to the accompanying Jupyter notebook, which we recommend to do
locally.

---

## Results

### BART

The results for evaluation on the dev dataset. training was done for 5 epochs.

|               | **Paraphrase Type Detection (acc)** | **Paraphrase Type Generation (BLEU)** |
|---------------|-------------------------------------|---------------------------------------|
| Baseline      | 0.833                               | 44.053                                |
| Improvement 1 | ...                                 | ...                                   |
| Improvement 2 | ...                                 | ...                                   |

### BERT

For BERT model, fine-tuning was done 2 times. For Multitask the model learned all 3 tasks one after another. For Independet the model learned tasks separately.

The results for the dev dataset.

| **Multitask** | **Sentiment Classification (acc)** | **Paraphrase Detection (acc)** | **Semantic Textual Similarity (cor)** |
|---------------|------------------------------------|--------------------------------|---------------------------------------|
| Baseline      | 0.515                              | 0.877                          | 0.849                                 |
| Improvement 1 | ...                                | ...                            | ...                                   |
| Improvement 2 | ...                                | ...                            | ...                                   |
| ...           | ...                                | ...                            | ...                                   |

Here Paraphrase Detection was trained for 1 epoch:

| **Independent** | **Sentiment Classification (acc)** | **Paraphrase Detection (acc)** | **Semantic Textual Similarity (cor)** |
|-----------------|------------------------------------|--------------------------------|---------------------------------------|
| Baseline        | 0.534                              | 0.860                          | 0.863                                 |
| Improvement 1   | ...                                | ...                            | ...                                   |
| Improvement 2   | ...                                | ...                            | ...                                   |
| ...             | ...                                | ...                            | ...                                   |

---

## Members Contribution

Explain the contribution of each group member:

**Daniel Ariza:**
- Phase 1:
  - Implemented the `embed` function in the `BertModel` class.
  - Implemented missing functionality for the sentiment analysis task.
  - Assisted in adding docstrings and type hints to functions in `bert.py` and `multitask_classifier.py`.
  - Filled and generated AI-usage card with input from all team members.
- Phase 2: ...

**Amirreza Aleyasin:**
- Phase 1:
  - Implemented the BART model for `paraphrase generation` and detection.
  - Implemented the `AdamW optimizer`.
- Phase 2: ...

**Pablo Jahnen:**
- Phase 1:
  - Implemented the `attention` function in the `BertSelfAttention` class.
  - Developed functionality for similarity prediction task.
  - Developed the training loop for similarity prediction task.
- Phase 2: 

**Ughur Mammadzada:**
- Phase 1:
  - Implemented the `BertLayer` class.
  - Developed functionality for paraphrase prediction task.
  - Developed the training loop for paraphrase prediction task.
- Phase 2: ...

**Enno Weber:**
- Phase 1: 
  - Implemented BART paraphrase detection transformer, training loop and test function.
- Phase 2: ...

---

## AI-Usage Card

Artificial Intelligence (AI) aided the development of this project. For transparency, we provide our [AI-Usage Card](./NeuralWordsmiths_AI_Usage_Card.pdf/) at the top.

## References

- [Project Description Deep Learning for Natural Language Processing University of Göttingen Teaching Staff](https://docs.google.com/document/d/1pZiPDbcUVhU9ODeMUI_lXZKQWSsxr7GO/edit): Lecturer: Dr. Terry Ruas, Teaching Assistants: Finn Schmidt, Niklas Bauer, Ivan Pluzhnikov, Jan Philip Wahle, Jonas Lührs
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805): Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762): Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
- [Paraphrase Types for Generation and Detection](https://aclanthology.org/2023.emnlp-main.746.pdf): Jan Philip Wahle, Bela Gipp, Terry Ruas, University of Göttingen, Germany {wahle,gipp,ruas}@uni-goettingen.de
- [SemEval-2016 Task 1: Semantic Textual Similarity, Monolingual and Cross-Lingual Evaluation](https://www.researchgate.net/publication/305334510_SemEval-2016_Task_1_Semantic_Textual_Similarity_Monolingual_and_Cross-Lingual_Evaluation): Eneko Agirre, Carmen Banea, Daniel Cer, Mona Diab
- [Context-aware Self-Attention Networks](https://arxiv.org/abs/1902.05766): Baosong Yang, Jian Li, Derek Wong, Lidia S. Chao, Xing Wang, Zhaopeng Tu

#TODO: (Phase 2) List all references (repositories, papers, etc.) used for your project.

## Acknowledgement

The project description, partial implementation, and scripts were adapted from the final project for the Stanford [CS 224N class](https://web.stanford.edu/class/cs224n/). The BERT implementation was adapted from the "minbert" assignment at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html). Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library. Scripts and code were modified by [Jan Philip Wahle](https://jpwahle.com/) and [Terry Ruas](https://terryruas.com/).

