import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from sacrebleu.metrics import BLEU
from torch import nn

from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig, BartModel
from torch.optim.lr_scheduler import StepLR

from optimizer import AdamW, SophiaG
#from utils import SwiGLU, GELU, SwiGLUFeedForward, RMSNorm, RotaryPositionalEmbedding, apply_rotary_pos_emb
#from typing import Optional, Tuple
from rouge_score import rouge_scorer

from utils import nums2word_word2nums, tag_pos

TQDM_DISABLE = False

# define the local path to the data
data_path = os.path.join(os.getcwd(), 'data')

config_dict = {
    "epochs": 10,
    "learning_rate": 1e-4,
    "optimizer": "SophiaG",
    "optimizer_params": {"lr": 1e-5, "betas": (0.1, 0.001), "eps": 1e-8, "weight_decay": 0.01},
    "use_scheduler": True,
    "scheduler_step_size": 1,
    "scheduler_gamma": 0.2,
    "batch_size": 16,
    "max_length": 256,
    "num_layers_to_freeze": 8,
    "dataset": "etpc-paraphrase-train.csv",
    "subset": 1,
    "val_dataset": "etpc-paraphrase-dev.csv",
    "test_dataset": "etpc-paraphrase-generation-test-student.csv",
    "penalized_bleu_epochs": [],
    "penalized_bleu_val": 0.0,
    "penalized_bleu_test": 0.0,
    "prefix": True,
    "prefix_length": 10,
    "prefix_method": "indirect",
    "use_gpu": True,
    "seed": 11711,
    "model": "facebook/bart-large",
    "tokenizer": "facebook/bart-large",
    "input_format": "sentence1",
    "target_format": "sentence2",
    "other_details": "",
}


def transform_data(dataset: pd.DataFrame, max_length: int = 256, batch_size: int = 1,
                   tokenizer_name: str = 'facebook/bart-large', shuffle: bool = False) -> DataLoader:
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
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    #scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    is_test = False
    if 'sentence2' not in dataset:
        is_test = True

    sentences = []
    target_sentences = []
    for _, row in dataset.iterrows():

        # TODO choose the most important tokens to mask using ROUGE score
        sentence1 = row['sentence1'] # input sentence
        sentence2 = row['sentence2'] if not is_test else None # target sentence

        # TODO converting numbers to words, this is to make sure the model does not hallucinate numbers
        #sentence1 = nums2word_word2nums(sentence1, input_type='digits', num_tag=False)

        # TODO tag the POS of the sentence
        sentence1_tokens, sentence1_tags = tag_pos(sentence1)
        ## must be removed ##
        # join the tokens and tags with "/"
        #sentence1 = [f"{token}/{tag}" for token, tag in zip(sentence1_tokens, sentence1_tags)]
        #taggged_sentence = ' '.join(sentence1)
        ## must be removed ##

        # TODO get the most important tokens
        #sentence1_tokens = row['sentence1_tokenized']
        #tokens = tokenizer.tokenize(sentence1)
        #important_tokens = get_important_tokens(sentence1, sentence2, scorer, tokens)

        # TODO mask the most important tokens
        #masked_sentence = mask_important_tokens(tokens, important_tokens, tokenizer)

        # TODO maks random verbs/adjectives/nouns/conjunctions
        # mask a random verb
        verb_tokens = [token for token, tag in zip(sentence1_tokens, sentence1_tags) if
                       tag == 'VERB']  # assumes there exists a tag for verbs
        if len(verb_tokens) > 0:
            # take a  verb randomly and mask
            verb_token = random.choice(verb_tokens)
            masked_sentence = sentence1.replace(verb_token, tokenizer.mask_token)
        else:
            masked_sentence = sentence1

        # convert conjugations to comma (conjugations has tag SCONJ)
        SCONJs = [token for token, tag in zip(sentence1_tokens, sentence1_tags) if tag == 'SCONJ']
        if len(SCONJs) > 0:
            masked_sentence = masked_sentence.replace(SCONJs[0], ',')

        # rotate the sentence parts if there is , in the between
        if ',' in masked_sentence:
            masked_sentence = masked_sentence.split(',')
            masked_sentence = masked_sentence[::-1]
            masked_sentence = ','.join(masked_sentence)

        # mask a random adjective
        adj_tokens = [token for token, tag in zip(sentence1_tokens, sentence1_tags) if
                      tag == 'ADJ']
        if len(adj_tokens) > 0:
            adj_token = random.choice(adj_tokens)
            masked_sentence = masked_sentence.replace(adj_token, tokenizer.mask_token)
        else:
            masked_sentence = masked_sentence

        # mask a random noun
        noun_tokens = [token for token, tag in zip(sentence1_tokens, sentence1_tags) if
                       tag == 'NOUN']
        if len(noun_tokens) > 0:
            noun_token = random.choice(noun_tokens)
            masked_sentence = masked_sentence.replace(noun_token, tokenizer.mask_token)


        sentence1_segment = ' '.join(map(str, eval(row['sentence1_segment_location'])))
        paraphrase_types = ' '.join(map(str, eval(row['paraphrase_types'])))
        formatted_sentence = f"{masked_sentence} {tokenizer.sep_token} {' '.join(sentence1_tags)}"
        #formatted_sentence = f"{masked_sentence} {tokenizer.sep_token} {sentence1_segment} {tokenizer.sep_token} {paraphrase_types}"
        #print("input: ", formatted_sentence)

        sentences.append(formatted_sentence)

        if not is_test:
            sentence2_segment = ' '.join(map(str, eval(row['sentence2_segment_location'])))
            formatted_sentence2 = f"{sentence2}"
            #print("target: ", formatted_sentence2)
            target_sentences.append(formatted_sentence2)


    # Tokenize the sentences
    inputs = tokenizer(sentences, max_length=max_length, padding=True, truncation=True, return_tensors="pt")

    if not is_test:
        labels = tokenizer(target_sentences, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
        dataset = TensorDataset(inputs.input_ids, inputs.attention_mask, labels.input_ids)
    else:
        dataset = TensorDataset(inputs.input_ids, inputs.attention_mask)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # log information
    config_dict["input_format"] = "{masked_sentence} {tokenizer.sep_token} {' '.join(sentence1_tags)}"
    config_dict["target_format"] = "{sentence2}"
    config_dict["other_details"] = "masked a random verb, adjective, noun, and conjunction"


    return data_loader

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

    # Freeze the specified number of encoder layers
    for i in range(num_layers_to_freeze):
        for param in model.model.encoder.layers[i].parameters():
            param.requires_grad = False

    return model


class PrefixModel(nn.Module):
    def __init__(self, base_model, prefix_length, prefix_method='direct'):
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

'''def transform_data(dataset: pd.DataFrame, max_length: int = 256, batch_size: int = 1,  tokenizer_name: str = 'facebook/bart-large', shuffle: bool = False) -> DataLoader:
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
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    is_test = False
    if 'sentence2' not in dataset:
        is_test = True

    sentences = []
    target_sentences = []
    for _, row in dataset.iterrows():
        sentence1 = row['sentence1']
        sentence2 = row['sentence2'] if not is_test else None

        sentence1_segment = ' '.join(map(str, eval(row['sentence1_segment_location'])))
        paraphrase_types = ' '.join(map(str, eval(row['paraphrase_types'])))
        formatted_sentence = f"{sentence1} {tokenizer.sep_token} {sentence1_segment} {tokenizer.sep_token} {paraphrase_types}"
        print("input: ", formatted_sentence)

        sentences.append(formatted_sentence)

        if not is_test:
            sentence2_segment = ' '.join(map(str, eval(row['sentence2_segment_location'])))
            formatted_sentence2 = f"{sentence2}"
            print("target: ", formatted_sentence2)
            target_sentences.append(formatted_sentence2)



    encodings = tokenizer(sentences, target_sentences, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
    target_encodings = tokenizer(target_sentences, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], target_encodings['input_ids'])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader
'''

def get_important_tokens(sentence1: str, sentence2: str, scorer: rouge_scorer.RougeScorer, tokens: list) -> list:
    """
    Get the most important tokens based on ROUGE score.
    """
    if sentence2 is None:
        # For test data, randomly select tokens as important
        return random.sample(tokens, k=min(5, len(tokens)))

    important_tokens = []
    for token in tokens:
        score = scorer.score(sentence1, sentence2)['rougeL'].fmeasure
        score_without_token = scorer.score(sentence1.replace(token, ''), sentence2)['rougeL'].fmeasure
        if score - score_without_token > 0.01:  # Threshold can be adjusted
            important_tokens.append(token)
    return important_tokens


def mask_important_tokens(tokens: list, important_tokens: list, tokenizer: AutoTokenizer) -> str:
    """
    Mask the important tokens in the sentence.
    """
    masked_tokens = [tokenizer.mask_token if token in important_tokens else token for token in tokens]
    return tokenizer.convert_tokens_to_string(masked_tokens)


def train_model(model: BartForConditionalGeneration,
                train_loader: DataLoader,
                val_loader: DataLoader,
                device: torch.device,
                tokenizer: AutoTokenizer,
                epochs: int = 5,
                learning_rate: float = 1e-5,
                optimizer_name: str = "AdamW",
                optimizer_params: dict = {"lr": 1e-5, "betas": (0.1, 0.001), "eps": 1e-8, "weight_decay": 0.01},
                use_scheduler: bool = False,
                scheduler_step_size: int = 1,
                scheduler_gamma: float = 0.2,
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

    # Prepare the optimizer
    if optimizer_name == "AdamW":
        optimizer = AdamW(model.parameters(), **optimizer_params)
    elif optimizer_name == "SophiaG":
        optimizer_params = {"lr": learning_rate, "betas": (0.965, 0.99), "rho": 0.04, "weight_decay": 1e-1}
        optimizer = SophiaG(model.parameters(), **optimizer_params)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Prepare the scheduler
    if use_scheduler:
        scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)


    # configured loss function
    loss_fc = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    def loss_value(outputs, labels):
        vocab_size = tokenizer.vocab_size
        return loss_fc(outputs.logits.view(-1, vocab_size), labels.view(-1))

    best_penalized_bleu = 0
    penalized_bleu_list = []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Train the model
        for batch in tqdm(train_loader, desc="Training"):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = loss_value(outputs, labels)
            #loss = outputs.loss
            loss.backward()
            optimizer.step()

            if use_scheduler:
                scheduler.step()

            print(f"Loss: {loss.item()}")

        # Evaluate the model with penalized BLEU score
        penalized_bleu = evaluate_model(model, val_loader, device, tokenizer)

        if penalized_bleu > best_penalized_bleu:
            best_penalized_bleu = penalized_bleu
            # Save the best model
            model.save_pretrained(output_dir)
            print(f"Model with score {best_penalized_bleu} saved.")


        # TODO using genetic algorithm to optimize the model with the best hyperparameters and objective best penalized BLEU score

        # log information
        # for each epoch, add penalized BLEU score like epoch 1: 0.5, epoch 2: 0.6, etc.
        penalized_bleu_list.append(penalized_bleu)

    # load the best model
    model = BartForConditionalGeneration.from_pretrained(output_dir)
    print(f"Best model loaded with penalized BLEU score: {best_penalized_bleu}")

    # log information
    config_dict["penalized_bleu_epochs"] = penalized_bleu_list
    config_dict["optimizer"] = optimizer_name

    return model

'''

def train_model(model: BartForConditionalGeneration,
                train_loader: DataLoader,
                val_dataset: Dataset,
                device: torch.device,
                tokenizer: AutoTokenizer,
                num_trials: int = 10,
                output_dir: str = "models/bart_finetuned_model"
                ) -> BartForConditionalGeneration:
    """
    Train the BART model using random hyperparameter search.
    """

    def train_with_params(epochs, learning_rate, step_size, gamma):
        model.train()
        optimizer = SophiaG(model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        loss_fc = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

        for epoch in range(epochs):
            for batch in train_loader:
                input_ids, attention_mask, labels = [x.to(device) for x in batch]
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = loss_fc(outputs.logits.view(-1, tokenizer.vocab_size), labels.view(-1))
                loss.backward()
                optimizer.step()
            scheduler.step()

            try:
                penalized_bleu = evaluate_model(model, val_dataset, device, tokenizer)
            except:
                penalized_bleu = 0

        return penalized_bleu

    best_score = float("-inf")
    best_params = None

    for trial in range(num_trials):
        # Random hyperparameter sampling
        epochs = 3
        learning_rate = random.uniform(1e-6, 1e-3)
        step_size = random.randint(1, 2)
        gamma = random.uniform(0.1, 0.9)

        print(f"Trial {trial + 1}/{num_trials}")
        print(f"Epochs: {epochs}, Learning rate: {learning_rate:.6f}, Step size: {step_size}, Gamma: {gamma:.3f}")

        # Train and evaluate with these hyperparameters
        score = train_with_params(epochs, learning_rate, step_size, gamma)

        print(f"Penalized BLEU Score: {score}")

        if score > best_score:
            best_score = score
            best_params = (epochs, learning_rate, step_size, gamma)

        print(f"Best score so far: {best_score}")
        print()

    print("Best hyperparameters found:")
    print(f"Epochs: {best_params[0]}")
    print(f"Learning rate: {best_params[1]:.6f}")
    print(f"Step size: {best_params[2]}")
    print(f"Gamma: {best_params[3]:.3f}")

    # Train the final model with the best hyperparameters
    final_model = train_with_params(*best_params)

    return final_model
'''


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
    model.eval()
    bleu = BLEU()
    predictions = []

    dataloader = transform_data(test_data, shuffle=False)
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

    #print("inputs: ", inputs)
    #print("references: ", references)
    #print("predictions: ", predictions)

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

    return penalized_bleu


def collect_logs(logs: dict, output_dir: str = "logs") -> None:
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
    # we split the train and generated dev, then usd dev as the validation set

    # subset for development
    frac = config_dict["subset"]
    train_dataset = train_dataset.sample(frac=frac)
    dev_dataset = dev_dataset.sample(frac=frac)
    test_dataset = test_dataset.sample(frac=frac)
    ###########################################################################

    train_loader = transform_data(train_dataset, shuffle=True, tokenizer_name=config_dict["tokenizer"], max_length=config_dict["max_length"], batch_size=config_dict["batch_size"])
    #dev_loader = transform_data(dev_dataset, shuffle=False, tokenizer_name=config_dict["tokenizer"], max_length=config_dict["max_length"], batch_size=confing_dict["batch_size"])
    test_loader = transform_data(test_dataset, shuffle=False, tokenizer_name=config_dict["tokenizer"], max_length=config_dict["max_length"], batch_size=config_dict["batch_size"])

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
                        scheduler_step_size=config_dict["scheduler_step_size"],
                        scheduler_gamma=config_dict["scheduler_gamma"])

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


## Optimization using genetic algorithm
# Mutation function
def mutate(config):
    """
    Randomly mutate a configuration parameter.
    """
    mutated_config = config.copy()

    # Randomly choose a parameter to mutate
    key = random.choice(list(mutated_config.keys()))

    if key == "learning_rate":
        mutated_config[key] = 10 ** random.uniform(-5, -2)
    elif key == "optimizer":
        mutated_config[key] = random.choice(["SophiaG", "AdamW"])
    elif key == "use_scheduler":
        mutated_config[key] = random.choice([True, False])
    elif key == "scheduler_step_size":
        mutated_config[key] = random.randint(1, 2)
    elif key == "scheduler_gamma":
        mutated_config[key] = random.uniform(0.1, 1.0)
    elif key == "batch_size":
        mutated_config[key] = random.randint(1, 10)
    elif key == "max_length":
        mutated_config[key] = random.randint(128, 512)
    elif key == "num_layers_to_freeze":
        mutated_config[key] = random.randint(0, 12)
    elif key == "prefix":
        mutated_config[key] = random.choice([True, False])
    elif key == "prefix_length":
        mutated_config[key] = random.randint(1, 20)
    elif key == "prefix_method":
        mutated_config[key] = random.choice(["indirect", "direct"])

    return mutated_config


# Crossover function
def crossover(parent1, parent2):
    """
    Perform crossover between two parent configurations.
    """
    child = {}
    for key in parent1.keys():
        child[key] = random.choice([parent1[key], parent2[key]])
    return child


# Genetic Algorithm
def genetic_algorithm(config_dict, population_size=10, generations=5, args=None):
    # Initialize population
    population = [mutate(config_dict) for _ in range(population_size)]

    for generation in range(generations):
        print(f"Generation {generation + 1}/{generations}")

        # Evaluate fitness of each individual
        fitness_scores = [(individual, finetune_paraphrase_generation(args, individual)) for individual in population]

        # Sort population by fitness (descending)
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        # Select the top half of the population as parents
        parents = [ind for ind, fit in fitness_scores[:population_size // 2]]

        # Create a new population through crossover and mutation
        new_population = parents.copy()
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population
        best_individual, best_fitness = fitness_scores[0]

        print(f"Generation {generation + 1} | Best Fitness: {best_fitness:.4f} | Best Config: {best_individual}")


    # run the best configuration to generate the paraphrases
    finetune_paraphrase_generation(args, best_individual)

    # save the best configuration
    collect_logs(best_individual, output_dir="logs/genetic_algorithm")

    return


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    # log information
    config_dict["seed"] = args.seed
    config_dict["use_gpu"] = args.use_gpu
    finetune_paraphrase_generation(args, config_dict)
    #genetic_algorithm(config_dict, population_size=3, generations=2, args=args)
