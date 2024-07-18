import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from sacrebleu.metrics import BLEU
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration

from optimizer import AdamW

TQDM_DISABLE = False

# define the local path to the data
data_path = os.path.join(os.getcwd(), 'data')


def transform_data(dataset: pd.DataFrame, max_length: int = 256, batch_size: int = 16,
                   tokenizer_name: str = 'facebook/bart-large') -> DataLoader:
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
        sentence1_segment = row['sentence1_segment_location']
        paraphrase_types = row['paraphrase_types']
        formatted_sentence = f"{sentence1} {tokenizer.sep_token} {sentence1_segment} {tokenizer.sep_token} {paraphrase_types}"
        sentences.append(formatted_sentence)

        if not is_test:
            sentence2 = row['sentence2']
            sentence2_segment = row['sentence2_segment_location']
            formatted_sentence2 = f"{sentence2} {tokenizer.sep_token} {sentence2_segment}"
            target_sentences.append(formatted_sentence2)

    # Tokenize the sentences
    inputs = tokenizer(sentences, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    if not is_test:
        labels = tokenizer(target_sentences, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        dataset = TensorDataset(inputs.input_ids, inputs.attention_mask, labels.input_ids)
    else:
        dataset = TensorDataset(inputs.input_ids, inputs.attention_mask)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader


def train_model(model: BartForConditionalGeneration,
                train_loader: DataLoader,
                val_loader: DataLoader,
                device: torch.device,
                tokenizer: AutoTokenizer,
                epochs: int = 3,
                learning_rate: float = 1e-5,
                output_dir: str = "bart_finetuned_model"
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
    optimizer = AdamW(model.parameters(), lr=learning_rate)

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
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            print(f"Loss: {loss.item()}")

        # Evaluate the model
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        print(f"Validation loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the model
            model.save_pretrained(output_dir)

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

    generated_sentences = []
    processed_ids = []

    with torch.no_grad():
        try:
            # Iterate over test data batches
            for i, batch in enumerate(tqdm(test_data, desc="Testing")):
                input_ids, attention_mask = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                # Generate paraphrases
                outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=256,
                                         num_beams=5, early_stopping=True)

                # Decode generated sentences
                decoded_sentences = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                generated_sentences.extend(decoded_sentences)

                # Add corresponding ids
                batch_size = len(decoded_sentences)
                if isinstance(test_ids, list) and i*batch_size < len(test_ids):
                    processed_ids.extend(test_ids[i*batch_size : (i+1)*batch_size])
                else:
                    processed_ids.extend(range(i*batch_size, (i+1)*batch_size))

        except Exception as e:
            print(f"An error occurred during model inference: {e}")

    # Ensure processed_ids and generated_sentences have the same length
    min_length = min(len(processed_ids), len(generated_sentences))
    processed_ids = processed_ids[:min_length]
    generated_sentences = generated_sentences[:min_length]

    # Create a DataFrame with 'id' and 'Generated_sentence2'
    result_df = pd.DataFrame({
        'id': processed_ids,
        'Generated_sentence2': generated_sentences
    })

    return result_df


def evaluate_model(model: BartForConditionalGeneration,
                   test_data: DataLoader,
                   device: torch.device,
                   tokenizer: AutoTokenizer
                   ) -> float:
    """
    Evaluate the model using the BLEU score.
    Args:
    model (BartForConditionalGeneration): The model to evaluate.
    test_data (DataLoader): DataLoader for test data.
    device (torch.device): Device to run the model on.
    tokenizer (AutoTokenizer): Tokenizer for the model.
    Returns:
    float: BLEU score of the model's performance.
    """
    model.eval()
    bleu = BLEU()
    predictions = []
    references = []

    with torch.no_grad():
        for batch in tqdm(test_data, desc="Evaluating"):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

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
            ref_text = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for g in labels
            ]

            predictions.extend(pred_text)
            references.extend([[r] for r in ref_text])

    model.train()

    # Calculate BLEU score
    bleu_score = bleu.corpus_score(predictions, references)
    return bleu_score.score


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
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

    train_dataset = pd.read_csv(f"{data_path}/etpc-paraphrase-train.csv", sep="\t")
    dev_dataset = pd.read_csv(f"{data_path}/etpc-paraphrase-dev.csv", sep="\t")
    test_dataset = pd.read_csv(f"{data_path}/etpc-paraphrase-generation-test-student.csv", sep="\t")

    # You might do a split of the train data into train/validation set here
    # we split the train and generated dev, then usd dev as the validation set

    train_data = transform_data(train_dataset)
    dev_data = transform_data(dev_dataset)
    test_data = transform_data(test_dataset)

    print(f"Loaded {len(train_dataset)} training samples.")

    model = train_model(model, train_data, dev_data, device, tokenizer)

    print("Training finished.")

    bleu_score = evaluate_model(model, dev_data, device, tokenizer)
    print(f"The BLEU-score of the model is: {bleu_score:.3f}")

    test_ids = test_dataset["id"]
    test_results = test_model(test_data, test_ids, device, model, tokenizer)
    test_results.to_csv(
        "predictions/bart/etpc-paraphrase-generation-test-output.csv", index=False, sep="\t"
    )


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_generation(args)
