import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from sacrebleu.metrics import BLEU
from torch import nn

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig, BartModel
from torch.optim.lr_scheduler import StepLR

from optimizer import AdamW, SophiaG
from utils import SwiGLU, GELU, SwiGLUFeedForward, RMSNorm, RotaryPositionalEmbedding, apply_rotary_pos_emb
from typing import Optional, Tuple
from rouge_score import rouge_scorer

TQDM_DISABLE = False

# define the local path to the data
data_path = os.path.join(os.getcwd(), 'data')


def transform_data(dataset: pd.DataFrame, max_length: int = 256, batch_size: int = 32,
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

        # TODO get the most important tokens
        #sentence1_tokens = row['sentence1_tokenized']
        #tokens = tokenizer.tokenize(sentence1)
        #important_tokens = get_important_tokens(sentence1, sentence2, scorer, tokens)


        # TODO mask the most important tokens
        #masked_sentence = mask_important_tokens(tokens, important_tokens, tokenizer)


        sentence1_segment = ' '.join(map(str, eval(row['sentence1_segment_location'])))
        paraphrase_types = ' '.join(map(str, eval(row['paraphrase_types'])))
        formatted_sentence = f"{sentence1} {tokenizer.sep_token} {sentence1_segment} {tokenizer.sep_token} {paraphrase_types}"
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

    return data_loader


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
                epochs: int = 10,
                learning_rate: float = 1e-5,
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
    #optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.1, 0.001), eps=1e-8, weight_decay=0.01)
    #optimizer = AdamW(model.parameters(), lr=learning_rate)
    optimizer = SophiaG(model.parameters(), lr=learning_rate)
    #scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

    # configured loss function
    loss_fc = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    def loss_value(outputs, labels):
        vocab_size = tokenizer.vocab_size
        return loss_fc(outputs.logits.view(-1, vocab_size), labels.view(-1))

    best_val_loss = float("inf")
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
            #loss = loss_value(outputs, labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            print(f"Loss: {loss.item()}")

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

    return penalized_bleu


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


def finetune_paraphrase_generation(args: argparse.Namespace) -> None:
    """
    Fine-tune the BART model for paraphrase generation.
    Args:
    args (argparse.Namespace): Command line arguments.
    """
    # clear cache
    torch.cuda.empty_cache()
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", local_files_only=True)
    # model = modified_BART_model()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)

    train_dataset = pd.read_csv(f"{data_path}/etpc-paraphrase-train.csv", sep="\t")
    dev_dataset = pd.read_csv(f"{data_path}/etpc-paraphrase-dev.csv", sep="\t")
    test_dataset = pd.read_csv(f"{data_path}/etpc-paraphrase-generation-test-student.csv", sep="\t")

    # You might do a split of the train data into train/validation set here
    # we split the train and generated dev, then usd dev as the validation set

    # subset for development
    train_dataset = train_dataset.sample(frac=0.001)
    dev_dataset = dev_dataset.sample(frac=0.001)
    test_dataset = test_dataset.sample(frac=0.001)
    ###########################################################################

    train_loader = transform_data(train_dataset, shuffle=True)
    dev_loader = transform_data(dev_dataset, shuffle=False)
    test_loader = transform_data(test_dataset, shuffle=False)

    print(f"Loaded {len(train_dataset)} training samples.")

    model = train_model(model, train_loader, dev_loader, device, tokenizer)

    print("Training finished.")

    bleu_score = evaluate_model(model, dev_dataset, device, tokenizer)
    print(f"The penalized BLEU-score of the model is: {bleu_score:.3f}")

    test_ids = test_dataset["id"]
    test_results = test_model(test_loader, test_ids, device, model, tokenizer)
    test_results.to_csv(
        "predictions/bart/etpc-paraphrase-generation-test-output.csv", index=False, sep="\t"
    )


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_generation(args)
