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


def transform_data(dataset, max_length=256, batch_size=32, tokenizer_name='facebook/bart-large'):
    """
    Turn the data to the format you want to use.
    Use AutoTokenizer to obtain encoding (input_ids and attention_mask).
    Tokenize the sentence pair in the following format:
    sentence_1 + SEP + sentence_1 segment location + SEP + paraphrase types.
    Return Data Loader.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Prepare the sentences
    sentences = []
    for _, row in dataset.iterrows():
        if 'sentence_1' not in row or 'segment_location' not in row or 'paraphrase_type' not in row:
            raise ValueError("Missing columns in the dataset. Required columns: 'sentence_1', 'segment_location', 'paraphrase_type'")
        
        sentence_1 = row['sentence_1']
        segment_location = row['segment_location']
        paraphrase_type = row['paraphrase_type']

        formatted_sentence = f"{sentence_1} {tokenizer.sep_token} {segment_location} {tokenizer.sep_token} {paraphrase_type}"
        sentences.append(formatted_sentence)

    # Tokenize the sentences
    encodings = tokenizer(sentences, truncation=True, padding=True, max_length=max_length, return_tensors='pt')

    # Create TensorDataset and DataLoader
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader
    raise NotImplementedError


def train_model(model, train_data, dev_data, device, tokenizer, epochs=3, learning_rate=5e-5, output_dir="output"):
    """
    Train the model. Return and save the model.
    """
    # Set model to training mode
    model.train()

    # Prepare the optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0

        for batch in tqdm(train_data, desc="Training"):
            # Move the batch to the device
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)

            # Clear any previously calculated gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

            # Accumulate the loss
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_data)
        print(f"Average Training Loss: {avg_epoch_loss:.4f}")

        # Evaluate on the development set
        model.eval()
        dev_loss = 0
        with torch.no_grad():
            for batch in tqdm(dev_data, desc="Evaluating"):
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss

                dev_loss += loss.item()

        avg_dev_loss = dev_loss / len(dev_data)
        print(f"Average Validation Loss: {avg_dev_loss:.4f}")

        # Save the model checkpoint
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model.save_pretrained(os.path.join(output_dir, f"checkpoint-epoch-{epoch + 1}"))
        tokenizer.save_pretrained(os.path.join(output_dir, f"checkpoint-epoch-{epoch + 1}"))

        model.train()  # Set model back to train mode

    # Save the final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model

    raise NotImplementedError


def test_model(test_data, test_ids, device, model, tokenizer):
    """
    Test the model. Generate paraphrases for the given sentences (sentence1) and return the results
    in form of a Pandas dataframe with the columns 'id' and 'Generated_sentence2'.
    The data format in the columns should be the same as in the train dataset.
    Return this dataframe.
    """
    # Set model to evaluation mode
    model.eval()

    generated_sentences = []

    with torch.no_grad():
        try:
            # Iterate over test data batches
            for batch in tqdm(test_data, desc="Testing"):
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)

                # Generate paraphrases
                outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=256, num_beams=5, early_stopping=True)

                # Decode generated sentences
                decoded_sentences = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                generated_sentences.extend(decoded_sentences)
        except Exception as e:
            print(f"An error occurred during model inference: {e}")

    # Create a DataFrame with 'id' and 'Generated_sentence2'
    result_df = pd.DataFrame({
        'id': test_ids,
        'Generated_sentence2': generated_sentences
    })

    return result_df
    raise NotImplementedError


def evaluate_model(model, test_data, device, tokenizer):
    """
    You can use your train/validation set to evaluate models performance with the BLEU score.
    """
    model.eval()
    bleu = BLEU()
    predictions = []
    references = []

    with torch.no_grad():
        for batch in test_data:
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


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()
    return args


def finetune_paraphrase_generation(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv", sep="\t")
    dev_dataset = pd.read_csv("data/etpc-paraphrase-dev.csv", sep="\t")
    test_dataset = pd.read_csv("data/etpc-paraphrase-generation-test-student.csv", sep="\t")

    # You might do a split of the train data into train/validation set here
    # ...

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
