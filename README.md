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
### local: 

To train the model, activate the environment and run:

```sh
python -u multitask_classifier.py --use_gpu
```

Important parameters and their descriptions can be seen by running:

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

## Methodology

We implemented the base BERT and BART for the first phase of the project.

#TODO hints: In this section, describe the process and methods used in the project. Briefly explain the ideas implemented to improve the model. Make sure to indicate how existing ideas were used and extended.

---

## Phase I

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

---

## Phase II

# Improvements upon Base Models

## 1. Proposals

### 1.1 Contextual Global Attention (CGA)
As part of the improvements for the sentiment analysis task of the project, Contextual Global Attention 
(CGA) was introduced to enhance BERT's performance on the sentiment analysis task and in the pooling of the 
encoded output embeddings. This alternate mechanism aims to enhance BERT's ability to capture long-range 
dependencies by  integrating global context into its self-attention mechanism. This shall enable BERT to make 
more informed decisions, building on empirical evidence that global context enhances self-attention networks.

### 1.2 Pooling Strategies
While the **CLS** token is traditionally used as the aggregate representation in BERT's output, it may 
not fully capture the semantic content of an entire sequence. To address this, alternative pooling 
strategies—such as **average pooling, max pooling, and attention-based pooling**—were tested. 
This experimentation aimed to identify whether these approaches could provide a more comprehensive 
representation, thereby improving the model's performance in sentiment analysis by better encapsulating 
the overall meaning of the input text.

### 1.3 Optimizer Choice
The choice of optimizer significantly impacts the training dynamics and final performance of models 
like BERT. To explore potential improvements, the newly developed **Sophia** optimizer was tested against 
the widely used **AdamW** optimizer. This comparison aimed to determine if Sophia could offer better 
convergence and performance, thereby optimizing the training process and enhancing the model’s 
effectiveness in sentiment analysis.

### BERT

### **1. Global Context Layer and Contextual Global Attention (CGA)**

To enhance BERT's ability to capture long-range dependencies, a **Global Context Layer** was integrated into the model. This layer computes a global context vector by averaging token embeddings and refining it through a feed-forward network. The refined context vector is incorporated into BERT’s self-attention mechanism via a custom **Contextual Global Attention (CGA)** mechanism, implemented in the `context_bert.py` file. The CGA mechanism introduces additional weight matrices and gating parameters that modulate the influence of the global context on token-level representations.

The **Contextual Global Attention (CGA)** was tested in three configurations: as an extra layer on top of the 12 stacked vanilla BERT layers, as a layer used for attention-based pooling, and in both configurations simultaneously.

**Mathematical Foundations:**

The integration of context in self-attention is defined by the following key formulae:

1. **Contextualized Query and Key Transformations**: 

$$
\begin{bmatrix}
\hat{\mathbf{Q}} \\
\hat{\mathbf{K}}
\end{bmatrix} = (1 - \begin{bmatrix} \lambda_Q \\ \lambda_K \end{bmatrix}) \begin{bmatrix} \mathbf{Q} \\ \mathbf{K} \end{bmatrix} + \begin{bmatrix} \lambda_Q \\ \lambda_K \end{bmatrix} \mathbf{C} \begin{bmatrix} \mathbf{U}_Q \\ \mathbf{U}_K \end{bmatrix}
$$

2. **Gating Mechanism for Contextual Influence**: 

\[
\begin{bmatrix}
\lambda_Q \\
\lambda_K
\end{bmatrix} = \sigma \left(\mathbf{Q} \mathbf{V}_Q^H + \mathbf{K} \mathbf{V}_K^H + \mathbf{C} \left[\mathbf{U}_Q \mathbf{V}_Q^C + \mathbf{U}_K \mathbf{V}_K^C \right]\right)
\]

3. **Output Representation**:

\[
\mathbf{O} = \text{ATT}(\mathbf{\hat{Q}}, \mathbf{\hat{K}})\mathbf{V}
\]

4. **Global Context Vector**:

\[
\mathbf{c} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{h}_i
\]

These formulae underpin the CGA mechanism, enhancing BERT’s ability to incorporate global context and improve performance on tasks requiring a comprehensive understanding of the input sequence.


---

## Experiments

### Learning all tasks vs. Learning one task:

- A BERT model was trained to be able to solve all 3 tasks, and was compared to a BERT model trained on the tasks independetly.
- The results for Sentiment Classification and Semantic Textual Similarity degrade, while for Paraphrase Detection increase.
- Most probable explanation: Dataset sizes are not equal. Later or bigger trainings degrade previous or smaller trainings.
- Possible solution: Trainings on bigger datasets first. Number of epochs relative to dataset size.

---

## Results

### BART

The results for evaluation on the dev dataset. training was done for 5 epochs.

| | **Paraphrase Type Detection (acc)** | **Paraphrase Type Generation (BLEU)** |
|----------|---------------|--------------|
| Baseline | 0.833 | 44.053 |
| Improvement 1 | ... | ... |
| Improvement 2 | ... | ... |

### BERT

For BERT model, fine-tuning was done 2 times. For Multitask the model learned all 3 tasks one after another. For Independet the model learned tasks separately.

The results for the dev dataset.

| **Multitask** | **Sentiment Classification (acc)** | **Paraphrase Detection (acc)** | **Semantic Textual Similarity (cor)** |
|----------|---------------|--------------|--------------|
| Baseline | 0.515 | 0.877 | 0.849 |
| Improvement 1 | ... | ... | ... |
| Improvement 2 | ... | ... | ... |
| ... | ... | ... | ... |


Here Paraphrase Detection was trained for 1 epoch:

| **Independent** | **Sentiment Classification (acc)** | **Paraphrase Detection (acc)** | **Semantic Textual Similarity (cor)** |
|----------|---------------|--------------|--------------|
| Baseline | 0.534 | 0.860 | 0.863 |
| Improvement 1 | ... | ... | ... |
| Improvement 2 | ... | ... | ... |
| ... | ... | ... | ... |


#TODO: Discuss your results, observations, correlations, etc.

---

## Hyperparameter Optimization

#TODO 

Briefly describe how you optimized your hyperparameters. If you focused strongly on hyperparameter optimization, include it in the Experiment section.

## Visualizations

#TODO

Add relevant graphs showing metrics like accuracy, validation loss, etc., during training. Compare different training processes of your improvements in these graphs.

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

#TODO: (Phase 2) List all references (repositories, papers, etc.) used for your project.

## Acknowledgement

The project description, partial implementation, and scripts were adapted from the final project for the Stanford [CS 224N class](https://web.stanford.edu/class/cs224n/). The BERT implementation was adapted from the "minbert" assignment at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html). Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library. Scripts and code were modified by [Jan Philip Wahle](https://jpwahle.com/) and [Terry Ruas](https://terryruas.com/).

