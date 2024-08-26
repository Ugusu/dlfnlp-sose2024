import argparse
import ast
import math
import os
import random

import pandas as pd
import torch

from sacrebleu.metrics import BLEU
from torch import nn

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration
from torch.optim.lr_scheduler import StepLR

from optimizer import AdamW, SophiaG

from utils import tag_pos, get_important_tokens

import multiprocessing
from functools import partial

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
from math import exp

TQDM_DISABLE = False

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# define the local path to the data
data_path = os.path.join(os.getcwd(), 'data')

config_dict = {
    "epochs": 3,
    "learning_rate": 3e-5,
    "optimizer": "SophiaG", # SophiaG or AdamW
    "optimizer_params": {"lr": 3e-5, "betas": (0.1, 0.001), "rho": 0.04, "weight_decay": 1e-1}, # for SophiaG optimizer_params = {"lr": 1e-5, "betas": (0.1, 0.001), "rho": 0.04, "weight_decay": 1e-1} # for AdamW {"lr": 1e-5, "betas": (0.1, 0.001), "eps": 1e-8, "weight_decay": 0.01}
    "use_scheduler": False,
    "scheduler_step_size": 1,
    "scheduler_gamma": 0.675,
    "batch_size": 64,
    "max_length": 256,
    "gradual_unfreezing": False,
    "num_layers_to_freeze": 0,
    "rl_weight": 0,
    "dataset": "etpc-paraphrase-train.csv",
    "subset": 1,
    "val_dataset": "etpc-paraphrase-dev.csv",
    "test_dataset": "etpc-paraphrase-generation-test-student.csv",
    "penalized_bleu_epochs": [],
    "penalized_bleu_val": 0.0,
    "penalized_bleu_test": 0.0,
    "prefix": False,
    "prefix_length": 10,
    "prefix_method": "indirect",
    "use_gpu": True,
    "seed": 11711,
    "model": "facebook/bart-large",
    "tokenizer": "facebook/bart-large",
    "input_format": "sentence1 {tokenizer.sep_token} {' '.join(sentence1_tags)}",
    "target_format": "sentence2",
    "other_details": "",
    "example_inputs": [],
    "example_references": [],
    "example_predictions": []
}


def process_row(row: pd.DataFrame, tokenizer_sep_token: str, tokenizer_mask_token: str, all_sentence1_tokens: list ,is_test: bool = False, is_eval: bool = False):
    """
    Process a row from the dataset to generate the input and target sentences for the model.
    Args:
    row (pd.Series): The row to process.
    tokenizer_sep_token (str): Tokenizer separator token.
    tokenizer_mask_token (str): Tokenizer mask token.
    is_test (bool): Whether the dataset is a test set. Defaults to False.
    Returns:
    Tuple[str, Optional[str]]: The input and target sentences.
    """

    formatted_sentence2 = None

    sentence1_tokens, sentence1_tags = tag_pos(row['sentence1'])

    # Masking operations
    masked_sentence, sentence1_tags_str = mask_tokens(row['sentence1'], sentence1_tokens, sentence1_tags, tokenizer_mask_token)

    least_freq_tokens = None
    if not is_test and not is_eval:
        # Get the most important tokens between the two sentences and add them to the training data, this ensures that the model learns to generate the most important tokens
        important_tokens = get_important_tokens(row['sentence1_tokenized'], row['sentence2_tokenized'], all_sentence1_tokens)
        least_freq_tokens = ' '.join(important_tokens)

        # Format input sentence for half of the training data using a coin flip
        if random.random() > 0.5:
            formatted_sentence1 = f"{masked_sentence} {tokenizer_sep_token} {sentence1_tags_str} {tokenizer_sep_token} {least_freq_tokens}"
        else:
            formatted_sentence1 = f"{masked_sentence} {tokenizer_sep_token} {sentence1_tags_str}"

    else:
        formatted_sentence1 = f"{masked_sentence} {tokenizer_sep_token} {sentence1_tags_str}"

    #print(formatted_sentence1)

    if not is_test:
        formatted_sentence2 = f"{row['sentence2']}"

    return formatted_sentence1, formatted_sentence2


def mask_tokens(sentence: str, tokens: list, tags: list, tokenizer_mask_token: str):
    """
    Mask random tokens but for certain parts of speech in the sentence.
    making a verb, adjective, and a noun
    converting a conjunction to a comma
    rotating the sentence parts if there is a comma in between
    Args:
    sentence (str): The sentence to mask.
    tokens (list): List of tokens in the sentence.
    tags (list): List of POS tags for the tokens.
    tokenizer_mask_token (str): Tokenizer mask token.
    Returns:
    Tuple[str, str]: The masked sentence and the sentence tags
    """

    # Mask a random verb
    verb_tokens = [token for token, tag in zip(tokens, tags) if tag == 'VERB']
    if verb_tokens:
        verb_token = random.choice(verb_tokens)
        # replace the token in tokens
        tokens = [tokenizer_mask_token if token == verb_token else token for token in tokens]
        #masked_sentence = masked_sentence.replace(verb_token, tokenizer_mask_token)

    # Convert a conjunction to comma
    sconjs = [token for token, tag in zip(tokens, tags) if tag == 'SCONJ']
    if sconjs:
        sconj = random.choice(sconjs)
        # replace the token in tokens
        tokens = [',' if token == sconj else token for token in tokens]
        # masked_sentence = masked_sentence.replace(sconjs[0], ',')

    # Rotate some of the sentences parts if there's a comma
    # order the tags to match the tokens by zipping them together
    # find the tag of the comma
    if ',' in sentence:
        punctuations = [token for token, tag in zip(tokens, tags) if token == ',']
        if punctuations:
            punctuation_token = random.choice(punctuations)
            punctuation_index = tokens.index(punctuation_token)
            # rotate sentence and tags based on the punctuation index
            # Rotate the masked sentence, tokens, and tags based on the punctuation index
            tokens = tokens[punctuation_index + 1:] + tokens[:punctuation_index + 1]
            tags = tags[punctuation_index + 1:] + tags[:punctuation_index + 1]


    # Mask a random adjective
    adj_tokens = [token for token, tag in zip(tokens, tags) if tag == 'ADJ']
    if adj_tokens:
        adj_token = random.choice(adj_tokens)
        # replace the token in tokens
        tokens = [tokenizer_mask_token if token == adj_token else token for token in tokens]
        #masked_sentence = masked_sentence.replace(adj_token, tokenizer_mask_token)

    # Mask a random noun
    noun_tokens = [token for token, tag in zip(tokens, tags) if tag == 'NOUN']
    if noun_tokens:
        noun_token = random.choice(noun_tokens)
        # replace the token in tokens
        tokens = [tokenizer_mask_token if token == noun_token else token for token in tokens]
        #masked_sentence = masked_sentence.replace(noun_token, tokenizer_mask_token)

    # join tokens to form the masked sentence
    masked_sentence = ' '.join(tokens)
    sentence_tags = ' '.join(tags)

    return masked_sentence, sentence_tags


def transform_data(dataset: pd.DataFrame, max_length: int = 256, batch_size: int = 1,
                   tokenizer_name: str = 'facebook/bart-large', shuffle: bool = False, is_eval = True) -> DataLoader:
    """
    Transform the dataset for model input. Tokenizes and formats data, returning a DataLoader.
    Args:
    dataset (pd.DataFrame): The dataset to transform.
    max_length (int): Maximum token length. Defaults to 256.
    batch_size (int): Size of data batches. Defaults to 16.
    tokenizer_name (str): Name of the tokenizer to use. Defaults to 'facebook/bart-large'.
    Returns:
    DataLoader: DataLoader containing the tokenized data.
    """

    print("Processing sentences...")
    #with joblib.Parallel(n_jobs=-1, backend="multiprocessing") as parallel:
    #    sentences, target_sentences = zip(*parallel(joblib.delayed(process_row)(row, tokenizer.sep_token, tokenizer.mask_token, 'sentence2' not in dataset) for _, row in dataset.iterrows()))

    # Initialize tokenizer outside the multiprocessing to avoid issues
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    sep_token = tokenizer.sep_token
    mask_token = tokenizer.mask_token
    is_test = 'sentence2' not in dataset

    # all sentence1 tokens
    all_sentence1_tokens = [ast.literal_eval(row["sentence1_tokenized"]) for _, row in dataset.iterrows()]
    merged_tokens = [token for sublist in all_sentence1_tokens for token in sublist]
    #print(all_sentence1_tokens)

    # Prepare the process_row function with fixed arguments
    process_row_partial = partial(process_row, tokenizer_sep_token=sep_token, tokenizer_mask_token=mask_token, all_sentence1_tokens=merged_tokens, is_test=is_test, is_eval=is_eval)

    # Use multiprocessing for parallel processing
    num_cores = multiprocessing.cpu_count() // 2
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = list(tqdm(
            pool.imap(process_row_partial, [row for _, row in dataset.iterrows()]),
            total=len(dataset),
            desc="Processing rows"
        ))

    sentences, target_sentences = zip(*results)

    print("Done processing sentences.")
    #print(sentences[0], target_sentences[0])
    #print(len(sentences), len(target_sentences))

    # Tokenize the sentences
    inputs = tokenizer(sentences, max_length=max_length, padding=True, truncation=True, return_tensors="pt")

    if target_sentences[0]:
        labels = tokenizer(target_sentences, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
        dataset = TensorDataset(inputs.input_ids, inputs.attention_mask, labels.input_ids)
    else:
        dataset = TensorDataset(inputs.input_ids, inputs.attention_mask)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # log information
    config_dict["input_format"] = "{masked_sentence} {tokenizer.sep_token} {' '.join(sentence1_tags)}"
    config_dict["target_format"] = "{sentence2}"
    config_dict["other_details"] = ("masked a random verb, adjective, noun, and conjunction \n"
                                    "rotated the sentence parts if there is , in between"
                                    "Adding the least frequent tokens between the two sentences to half of the training data"
                                    "This is to make sure the model learns to generate the most important tokens, and also learn"
                                    "that htis type of input may be None, as it is for the validation and test")

    return data_loader


class OutputStrippingLayer(torch.nn.Module):
    """
    A custom layer that strips unnecessary tokens from the generated output.
    This layer can be placed after the decoder in the BART model.
    """

    def __init__(self, tokenizer):
        super(OutputStrippingLayer, self).__init__()
        self.tokenizer = tokenizer

    def forward(self, generated_ids):
        # Convert IDs to tokens
        tokens = [self.tokenizer.decode(g, skip_special_tokens=True).strip() for g in generated_ids]

        # Strip specific patterns if necessary (e.g., POS tags)
        # This is a placeholder: you might want to customize this
        stripped_tokens = [self.strip_output(t) for t in tokens]

        # Convert tokens back to IDs
        stripped_ids = [self.tokenizer.encode(t, return_tensors="pt")[0] for t in stripped_tokens]

        return stripped_ids

    def strip_output(self, sentence):
        # Placeholder for stripping logic, e.g., remove certain POS tags or patterns
        # Customize this function based on your use case
        # Example: strip extra commas or specific tokens
        stripped_sentence = sentence.replace(" ,", ",")  # Example of simple stripping
        return stripped_sentence


def modified_BART_model(num_layers_to_freeze: int = 8):
    """
    Modifies a BARTForConditionalGeneration model for paraphrasing by freezing some encoder layers.

    Args:
    model (BartForConditionalGeneration): The pre-trained BART model
    num_layers_to_freeze (int): Number of encoder layers to freeze (default: 8)

    Returns:
    BartForConditionalGeneration: The modified BART model
    """
    #config = BartConfig.from_pretrained("facebook/bart-large", local_files_only=True)
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)

    # Freeze the specified number of encoder layers
    for i in range(num_layers_to_freeze):
        for param in model.model.encoder.layers[i].parameters():
            param.requires_grad = False

    for i in range(num_layers_to_freeze):
        for param in model.model.decoder.layers[i].parameters():
            param.requires_grad = False

    # Add a custom layer for stripping the output
    stripping_layer = OutputStrippingLayer(tokenizer)
    model.stripping_layer = stripping_layer

    # Calculate the number of trainable layers
    num_trainable = 0
    for layer in model.model.encoder.layers:
        trainable_params = sum([1 for param in layer.parameters() if param.requires_grad])
        if trainable_params > 0:
            num_trainable += 1

    num_trainable_d = 0
    for layer in model.model.decoder.layers:
        trainable_params = sum([1 for param in layer.parameters() if param.requires_grad])
        if trainable_params > 0:
            num_trainable_d += 1

    print("Number of trainable encoder-decoder layers:", num_trainable, '-', num_trainable_d)

    # log information
    config_dict["other_details"].join("Modified BART model for paraphrasing by adding a custom layer for stripping the output")

    return model


def gradual_unfreezing(model, num_layers_to_unfreeze: int = 2, max_layers: int = 8):
    """
    Gradually unfreezes the encoder layers of a BART model.

    Args:
    model (BartForConditionalGeneration): The pre-trained BART model
    num_layers_to_unfreeze (int): Number of encoder layers to unfreeze (default: 2)
    does not work with PreFixModel
    """

    if isinstance(model, PrefixModel):
        model.gradual_unfreezing(num_layers_to_unfreeze, max_layers)

    else:
        # Calculate the number of trainable layers
        num_trainable = 0
        for layer in model.model.encoder.layers:
            trainable_params = sum([1 for param in layer.parameters() if param.requires_grad])
            if trainable_params > 0:
                num_trainable += 1

        num_trainable_d = 0
        for layer in model.model.decoder.layers:
            trainable_params = sum([1 for param in layer.parameters() if param.requires_grad])
            if trainable_params > 0:
                num_trainable_d += 1

        if num_trainable == max_layers:
            print(f"limit for unfreezing reached, {max_layers} layers are already unfrozen")
            config_dict["other_details"].join(f"limit for unfreezing reached, {max_layers} layers are already unfrozen")
            return model

        # Unfreeze the specified number of last layers in encoder and decoder
        for i in range(num_layers_to_unfreeze):
            for param in model.model.encoder.layers[-i].parameters():
                param.requires_grad = True

        for i in range(num_layers_to_unfreeze):
            for param in model.model.decoder.layers[-i].parameters():
                param.requires_grad = True

        print("Number of trainable encoder-decoder layers:", num_trainable + num_layers_to_unfreeze, '-', num_trainable_d + num_layers_to_unfreeze)

    return model


class PrefixModel(nn.Module):
    """
    PrefixModel class that adds a prefix to the input sequence before passing it to a BART model.
    This helps the model to learn to generate a target sequence based on the prefix (finetuning on prefix) for specific task, here paraphrasing.
    """
    def __init__(self, base_model, prefix_length, prefix_method='indirect'):
        super().__init__()
        self.base_model = base_model
        self.prefix_length = prefix_length
        self.prefix_method = prefix_method

        # Initialize the prefix
        self.prefix = nn.Parameter(torch.randn(1, prefix_length, base_model.config.d_model))

        if prefix_method == 'indirect':
            # For indirect method, we'll use an MLP to generate the prefix
            self.prefix_mlp = nn.Sequential(
                nn.Linear(base_model.config.d_model, base_model.config.d_model),
                nn.ReLU(),
                nn.Linear(base_model.config.d_model, base_model.config.d_model)
            )

    def get_prefix(self, batch_size):
        if self.prefix_method == 'direct':
            return self.prefix.expand(batch_size, -1, -1)
        elif self.prefix_method == 'indirect':
            # Generate prefix through MLP
            prefix = self.prefix.expand(batch_size, -1, -1)
            return self.prefix_mlp(prefix)

    def gradual_unfreezing(self, num_layers_to_unfreeze: int = 2, max_layers: int = 8):
        """
        Gradually unfreezes the encoder and decoder layers of the base model.

        Args:
        num_layers_to_unfreeze (int): Number of encoder and decoder layers to unfreeze (default: 2)
        """
        # Calculate the number of trainable layers
        num_trainable = 0
        for layer in self.base_model.model.encoder.layers:
            trainable_params = sum([1 for param in layer.parameters() if param.requires_grad])
            if trainable_params > 0:
                num_trainable += 1

        num_trainable_d = 0
        for layer in self.base_model.model.decoder.layers:
            trainable_params = sum([1 for param in layer.parameters() if param.requires_grad])
            if trainable_params > 0:
                num_trainable_d += 1

        if num_trainable == max_layers:
            print(f"limit for unfreezing reached, {max_layers} layers are already unfrozen")
            config_dict["other_details"].join(f"limit for unfreezing reached, {max_layers} layers are already unfrozen")
            return

        # Unfreeze the specified number of last layers in encoder and decoder
        for i in range(num_layers_to_unfreeze):
            for param in self.base_model.model.encoder.layers[-i].parameters():
                param.requires_grad = True

        for i in range(num_layers_to_unfreeze):
            for param in self.base_model.model.decoder.layers[-i].parameters():
                param.requires_grad = True

        # Calculate the number of trainable layers
        num_trainable = 0
        for layer in self.base_model.model.encoder.layers:
            trainable_params = sum([1 for param in layer.parameters() if param.requires_grad])
            if trainable_params > 0:
                num_trainable += 1

        num_trainable_d = 0
        for layer in self.base_model.model.decoder.layers:
            trainable_params = sum([1 for param in layer.parameters() if param.requires_grad])
            if trainable_params > 0:
                num_trainable_d += 1

        print("Number of trainable encoder layers:", num_trainable, '-', num_trainable_d)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, labels=None):
        batch_size = input_ids.shape[0]

        # Get the prefix
        prefix = self.get_prefix(batch_size)

        # Get the embeddings from the base model
        inputs_embeds = self.base_model.model.encoder.embed_tokens(input_ids)

        # Concatenate the prefix with the input embeddings
        inputs_embeds = torch.cat([prefix, inputs_embeds], dim=1)

        # Adjust the attention mask to account for the prefix
        if attention_mask is not None:
            prefix_attention_mask = torch.ones(batch_size, self.prefix_length, device=attention_mask.device)
            attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)

        # Forward pass through the base model
        outputs = self.base_model(inputs_embeds=inputs_embeds,
                                  attention_mask=attention_mask,
                                  decoder_input_ids=decoder_input_ids,
                                  labels=labels)

        return outputs

    def save_pretrained(self, output_dir):
        self.base_model.save_pretrained(output_dir)

    def generate(self, input_ids, attention_mask=None, max_length=None, num_beams=None, early_stopping=None):
        batch_size = input_ids.shape[0]

        # Get the prefix
        prefix = self.get_prefix(batch_size)

        # Get the embeddings from the base model
        inputs_embeds = self.base_model.model.encoder.embed_tokens(input_ids)

        # Concatenate the prefix with the input embeddings
        inputs_embeds = torch.cat([prefix, inputs_embeds], dim=1)

        # Adjust the attention mask to account for the prefix
        if attention_mask is not None:
            prefix_attention_mask = torch.ones(batch_size, self.prefix_length, device=attention_mask.device)
            attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)

        # Forward pass through the base model
        outputs = self.base_model.generate(inputs_embeds=inputs_embeds,
                                           attention_mask=attention_mask,
                                           max_length=max_length,
                                           num_beams=num_beams,
                                           early_stopping=early_stopping)

        return outputs

def compute_bleu_like_score(reference_tokens, generated_tokens):
    """
    Compute a BLEU-like score manually.
    Args:
    reference_tokens (list): List of tokens in the reference sentence.
    generated_tokens (list): List of tokens in the generated sentence.
    Returns:
    float: The BLEU-like score.
    """
    reference_counter = Counter(reference_tokens)
    generated_counter = Counter(generated_tokens)

    # Calculate precision
    overlap = sum(min(generated_counter[word], reference_counter[word]) for word in generated_tokens)
    precision = overlap / len(generated_tokens) if len(generated_tokens) > 0 else 0

    # Calculate brevity penalty
    ref_len = len(reference_tokens)
    gen_len = len(generated_tokens) if len(generated_tokens) > 0 else 1
    brevity_penalty = exp(1 - ref_len / gen_len) if gen_len < ref_len else 1.0

    # Compute BLEU-like score
    bleu_like_score = brevity_penalty * precision
    return bleu_like_score


def cosine_similarity(sentence1, sentence2):
    """Calculate the cosine similarity between two sentences."""
    # Tokenize the sentences into words
    words1 = sentence1.lower().split()
    words2 = sentence2.lower().split()

    # Create word frequency dictionaries
    freq1 = Counter(words1)
    freq2 = Counter(words2)

    # Get the set of unique words from both sentences
    unique_words = set(freq1.keys()) | set(freq2.keys())

    # Calculate dot product and magnitudes
    dot_product = sum(freq1.get(word, 0) * freq2.get(word, 0) for word in unique_words)
    magnitude1 = math.sqrt(sum(freq1.get(word, 0)**2 for word in unique_words))
    magnitude2 = math.sqrt(sum(freq2.get(word, 0)**2 for word in unique_words))

    # Avoid division by zero
    if magnitude1 * magnitude2 == 0:
        return 0

    return dot_product / (magnitude1 * magnitude2)

# adopting Reinforcement Learning for Paraphrase Generation
def compute_reward(generated_sentence: str, reference: str, input_sentence: str, tokenizer):
    """
    Compute the reward for a generated sentence based on a BLEU-like score and input sentence similarity.
    Args:
    generated_sentence (str): The generated sentence.
    reference (str): The reference sentence.
    input_sentence (str): The input sentence.
    tokenizer (AutoTokenizer): Tokenizer for the model.
    Returns:
    float: The computed reward.
    """

    # Tokenize sentences
    reference_tokens = reference.split()
    generated_tokens = generated_sentence.split()

    # Compute BLEU-like score
    bleu_like_score = compute_bleu_like_score(reference_tokens, generated_tokens)

    # Compute cosine similarity between generated sentence and input sentence
    similarity_score = cosine_similarity(reference, generated_sentence)

    # Compute reward
    reward = 0.50 * bleu_like_score + 0.50 * similarity_score

    #print(f"BLEU-like score: {bleu_like_score}, Similarity score: {similarity_score}, Reward: {reward}")

    return reward


def decode_output(outputs, tokenizer):
    """
    Decode the output of the model.
    """
    generated_paraphrases = []
    paraphrases = [
        tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for g in outputs
    ]
    generated_paraphrases.extend(paraphrases)

    return generated_paraphrases


def train_model(model: BartForConditionalGeneration,
                train_loader: DataLoader,
                val_data: pd.DataFrame,
                device: torch.device,
                tokenizer: AutoTokenizer,
                epochs: int = 5,
                learning_rate: float = 1e-5,
                optimizer_name: str = "AdamW",
                optimizer_params: dict = {"lr": 1e-5, "betas": (0.1, 0.001), "eps": 1e-8, "weight_decay": 0.01},
                use_scheduler: bool = False,
                gradual_unfreeze: bool = False,
                scheduler_step_size: int = 1,
                scheduler_gamma: float = 0.2,
                rl_weight: float = 0.5,
                output_dir: str = "models/bart_finetuned_model"
                ) -> BartForConditionalGeneration:
    """
    Train the BART model. Save and return the best model based on validation loss.
    Args:
    model (BartForConditionalGeneration): The model to train.
    train_loader (DataLoader): DataLoader for training data.
    val_loader (DataLoader): DataLoader for validation data.
    device (torch.device): Device to train the model on.
    tokenizer (AutoTokenizer): Tokenizer for the model.
    epochs (int): Number of training epochs. Defaults to 3.
    learning_rate (float): Learning rate for the optimizer. Defaults to 1e-5.
    output_dir (str): Directory to save the trained model. Defaults to "bart_finetuned_model".
    Returns:
    BartForConditionalGeneration: The trained model.
    """
    # Set model to training mode
    model.train()

    try:
        num_layers = model.config.encoder_layers
        num_trainable_layers = 0
    except:
        num_layers = 12
        num_trainable_layers = 0

    # Prepare the optimizer
    optimizer_params["lr"] = learning_rate
    if optimizer_name == "AdamW":
        optimizer = AdamW(model.parameters(), **optimizer_params)
    elif optimizer_name == "SophiaG":
        optimizer = SophiaG(model.parameters(), **optimizer_params)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Prepare the scheduler
    if use_scheduler:
        scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)


    # configured loss function
    loss_fc = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    def loss_value(outputs, labels, rewards=None):
        vocab_size = tokenizer.vocab_size
        if rewards is None:
            return loss_fc(outputs.logits.view(-1, vocab_size), labels.view(-1))
        else:
            rewards = torch.tensor(rewards, device=device)
            rl_loss = loss_fc(outputs.view(-1, vocab_size), labels.view(-1))
            #print(f"RL loss: {rl_loss}")
            # lower loss is better, higher reward is better
            rl_loss = rl_loss - (rl_loss * rewards.mean())
            #print(f"RL loss: {rl_loss}")
            return rl_loss


    best_penalized_bleu = 0
    best_loss = float("inf")
    penalized_bleu_list = []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        if gradual_unfreeze and epoch > 0: # using no trainable layers for the first epoch to train the first PrefixModel with pretrained BART
            # more epochs > fewer layers to unfreeze
            if epochs <= num_layers:
                num_layers_to_unfreeze = num_layers//epochs
                if num_trainable_layers > num_layers:
                    num_layers_to_unfreeze = 0
                    num_trainable_layers = num_layers
                num_trainable_layers += num_layers_to_unfreeze
            model = gradual_unfreezing(model, num_layers_to_unfreeze= num_trainable_layers)

        total_loss = 0
        # Train the model
        for batch in tqdm(train_loader, desc="Training"):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Supervised learning step
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            sl_loss = loss_value(outputs, labels)
            #sl_loss = outputs.loss

            # Reinforcement learning step
            with torch.no_grad():
                generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=50, num_beams=5, early_stopping=True)
            generated_paraphrases = decode_output(generated_ids, tokenizer)
            reference_paraphrases = decode_output(labels, tokenizer)
            input_sentences = decode_output(input_ids, tokenizer)
            rewards = [compute_reward(gen, ref, inp, tokenizer) for gen, ref, inp in zip(generated_paraphrases, reference_paraphrases, input_sentences)]

            # Policy gradient update
            # If the model returns a tuple or custom object, extract the relevant tensor
            if isinstance(outputs, tuple):
                rl_outputs = outputs[0]  # Assume the first element is the logits
            elif hasattr(outputs, 'logits'):
                rl_outputs = outputs.logits

            rl_loss = loss_value(rl_outputs, labels, rewards)

            loss = (1 - rl_weight) * sl_loss + rl_weight * rl_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            #print(f"batch loss: {loss}")

        total_loss /= len(train_loader)

        if use_scheduler:
            print(f"scheduler step, learning rate: {optimizer.param_groups[0]['lr']}")
            scheduler.step()

        print(f"Loss: {total_loss}")

        try:
            # Evaluate the model with penalized BLEU score
            penalized_bleu = evaluate_model(model, val_data, device, tokenizer)
        except:
            print("Failed to evaluate model. Probably the output is None.")
            penalized_bleu = 0

        #if penalized_bleu > best_penalized_bleu:
        #    best_penalized_bleu = penalized_bleu
        #    # Save the best model
        #    model.save_pretrained(output_dir)
        #    print(f"Model with score {best_penalized_bleu} saved.")

        #if total_loss < best_loss:
        #    best_loss = total_loss
        #    # Save the best model
        #    model.save_pretrained(output_dir)
        #    print(f"Model with loss {best_loss} saved.")

        # choose the best model with both highest penalized BLEU score with a tolerance and lowest loss - Multi-objective optimization
        tolerance = 0.9
        if penalized_bleu >= best_penalized_bleu - tolerance and total_loss <= best_loss:
            best_penalized_bleu = penalized_bleu
            best_loss = total_loss
            # Save the best model
            model.save_pretrained(output_dir)
            print(f"Model with score {best_penalized_bleu} and loss {best_loss} saved.")

        # log information
        # for each epoch, add penalized BLEU score like epoch 1: 0.5, epoch 2: 0.6, etc.
        penalized_bleu_list.append(penalized_bleu)

    # load the best model
    model = BartForConditionalGeneration.from_pretrained(output_dir, local_files_only=True)
    #print(f"Best model loaded with penalized BLEU score: {best_penalized_bleu}")
    print(f"Best model loaded with lowest loss: {best_loss}")

    # log information
    config_dict["penalized_bleu_epochs"] = penalized_bleu_list
    config_dict["optimizer"] = optimizer_name
    config_dict['other_details'].join("saving the best model with the highest penalized BLEU score and lowest loss together \n"
                                      "using reinforcement learning with a weight of 0.8 for the RL loss")

    return model



def test_model(test_data: DataLoader,
               test_ids: pd.Series,
               device: torch.device,
               model: BartForConditionalGeneration,
               tokenizer: AutoTokenizer
               ) -> pd.DataFrame:
    """
    Test the model by generating paraphrases for the given sentences and return the results in a DataFrame.
    Args:
    test_data (DataLoader): DataLoader for test data.
    test_ids (pd.Series): Series of test data IDs.
    device (torch.device): Device to run the model on.
    model (BartForConditionalGeneration): The model to test.
    tokenizer (AutoTokenizer): Tokenizer for the model.
    Returns:
    pd.DataFrame: DataFrame containing the test IDs and generated paraphrases.
    """
    # Set model to evaluation mode
    model.eval()
    generated_paraphrases = []

    with torch.no_grad():
        for batch in tqdm(test_data, desc="Testing"):
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Generate paraphrases
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=5,
                early_stopping=True,
            )

            paraphrases = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for g in outputs
            ]
            generated_paraphrases.extend(paraphrases)

    results_df = pd.DataFrame({"id": test_ids, "Generated_sentence2": generated_paraphrases})
    return results_df


def evaluate_model(model, test_data, device, tokenizer):
    """
    You can use your train/validation set to evaluate models performance with the BLEU score.
    test_data is a Pandas Dataframe, the column "sentence1" contains all input sentence and
    the column "sentence2" contains all target sentences
    """
    model.to(device)
    model.eval()
    bleu = BLEU()
    predictions = []

    dataloader = transform_data(test_data, shuffle=False, is_eval=True)
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Generate paraphrases
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=5,
                early_stopping=True,
            )

            pred_text = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for g in outputs
            ]

            predictions.extend(pred_text)


    inputs = test_data["sentence1"].tolist()
    references = test_data["sentence2"].tolist()

    print("inputs: ", inputs[0])
    print("references: ", references[0])
    print("predictions: ", predictions[0])

    model.train()
    # Calculate BLEU score
    bleu_score_reference = bleu.corpus_score(references, [predictions]).score
    # Penalize BLEU score if its to close to the input
    bleu_score_inputs = 100 - bleu.corpus_score(inputs, [predictions]).score

    print(f"BLEU Score: {bleu_score_reference}", f"Negative BLEU Score with input: {bleu_score_inputs}")

    # Penalize BLEU and rescale it to 0-100
    # If you perfectly predict all the targets, you should get a penalized BLEU score of around 52
    penalized_bleu = bleu_score_reference * bleu_score_inputs / 52
    print(f"Penalized BLEU Score: {penalized_bleu}")

    # log information
    config_dict["penalized_bleu_val"] = penalized_bleu
    config_dict["example_inputs"] = inputs[:5]
    config_dict["example_references"] = references[:5]
    config_dict["example_predictions"] = predictions[:5]

    return penalized_bleu


def collect_logs(logs: dict, output_dir: str = "logs/bart_generation") -> None:
    """
    Collect logs and save them to a file.
    Args:
    logs (dict): Dictionary containing logs.
    """
    os.makedirs(output_dir, exist_ok=True)
    # assign log_file nam with ordinal number
    log_file = f"{output_dir}/log_{len(os.listdir(output_dir)) + 1}.txt"
    with open(log_file, "w") as f:
        for key, value in logs.items():
            f.write(f"{key}: {value}\n")

    print(f"Logs saved to {log_file}")

    return

def seed_everything(seed: int = 11711) -> None:
    """
    Set random seed for reproducibility.
    Args:
    seed (int): Seed value. Defaults to 11711.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    Returns:
    argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()
    return args


def finetune_paraphrase_generation(args: argparse.Namespace, config_dict: dict) -> None:
    """
    Fine-tune the BART model for paraphrase generation.
    Args:
    args (argparse.Namespace): Command line arguments.
    """
    # clear cache
    torch.cuda.empty_cache()

    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    #model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", local_files_only=True)
    model = modified_BART_model(num_layers_to_freeze=config_dict["num_layers_to_freeze"])
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)

    # Create PrefixModel
    if config_dict["prefix"]:
        model = PrefixModel(model, config_dict["prefix_length"], config_dict["prefix_method"])
        model.to(device)

    train_dataset = pd.read_csv(f"{data_path}/etpc-paraphrase-train.csv", sep="\t")
    dev_dataset = pd.read_csv(f"{data_path}/etpc-paraphrase-dev.csv", sep="\t")
    test_dataset = pd.read_csv(f"{data_path}/etpc-paraphrase-generation-test-student.csv", sep="\t")

    # You might do a split of the train data into train/validation set here
    ## we split the train and generated dev, then usd dev as the validation set

    # subset for development
    frac = config_dict["subset"]
    train_dataset = train_dataset.sample(frac=frac)
    dev_dataset = dev_dataset.sample(frac=frac)
    test_dataset = test_dataset.sample(frac=frac)
    ###########################################################################

    train_loader = transform_data(train_dataset, shuffle=True, tokenizer_name=config_dict["tokenizer"], max_length=config_dict["max_length"], batch_size=config_dict["batch_size"], is_eval=False)
    #dev_loader = transform_data(dev_dataset, shuffle=False, tokenizer_name=config_dict["tokenizer"], max_length=config_dict["max_length"], batch_size=confing_dict["batch_size"], is_eval=True)
    test_loader = transform_data(test_dataset, shuffle=False, tokenizer_name=config_dict["tokenizer"], max_length=config_dict["max_length"], batch_size=config_dict["batch_size"], is_eval=True)

    print(f"Loaded {len(train_dataset)} training samples.")

    model = train_model(model,
                        train_loader,
                        dev_dataset,
                        device,
                        tokenizer,
                        epochs=config_dict["epochs"],
                        optimizer_name=config_dict["optimizer"],
                        optimizer_params=config_dict["optimizer_params"],
                        learning_rate=config_dict["learning_rate"],
                        use_scheduler=config_dict["use_scheduler"],
                        gradual_unfreeze=config_dict["gradual_unfreezing"],
                        scheduler_step_size=config_dict["scheduler_step_size"],
                        scheduler_gamma=config_dict["scheduler_gamma"],
                        rl_weight=config_dict["rl_weight"],
                        )

    print("Training finished.")

    bleu_score = evaluate_model(model, dev_dataset, device, tokenizer)
    print(f"The penalized BLEU-score of the model is: {bleu_score:.3f}")

    test_ids = test_dataset["id"]
    test_results = test_model(test_loader, test_ids, device, model, tokenizer)
    test_results.to_csv(
        "predictions/bart/etpc-paraphrase-generation-test-output.csv", index=False, sep="\t"
    )

    collect_logs(config_dict)

    return bleu_score



if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    # log information
    config_dict["seed"] = args.seed
    config_dict["use_gpu"] = args.use_gpu
    finetune_paraphrase_generation(args, config_dict)

