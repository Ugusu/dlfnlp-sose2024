from torch.optim.lr_scheduler import StepLR
import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from sacrebleu.metrics import BLEU
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration

from optimizer import AdamW

TQDM_DISABLE = False

# define the local path to the data
data_path = os.path.join(os.getcwd(), 'data')


def transform_data(dataset, max_length=256, batch_size=16, tokenizer_name='facebook/bart-large'):
    """
        Turn the data to the format you want to use.
        Use AutoTokenizer to obtain encoding (input_ids and attention_mask).
        Tokenize the sentence pair in the following format:
        sentence_1 + SEP + sentence_1 segment location + SEP + paraphrase types.
        Return Data Loader.
        """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    is_test = False
    if 'sentence2' not in dataset:
        is_test = True

    sentences = []
    target_sentences = []
    for row in dataset:
        sentence1 = dataset['sentence1']
        sentence1_segment = dataset['sentence1_segment_location']
        paraphrase_types = dataset['paraphrase_types']
        formatted_sentence = sentence1 + tokenizer.sep_token + sentence1_segment + tokenizer.sep_token + paraphrase_types
        sentences.extend(formatted_sentence)

    if not is_test:
        for row in dataset:
            sentence1 = dataset['sentence2']
            sentence1_segment = dataset['sentence2_segment_location']
            paraphrase_types = dataset['paraphrase_types']
            formatted_sentence = sentence1 + tokenizer.sep_token + sentence1_segment + tokenizer.sep_token + paraphrase_types
            target_sentences.extend(formatted_sentence)

    encodings = tokenizer(sentences, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
    if is_test:
        dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataloader

    target_encodings = tokenizer(target_sentences, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
    labels = target_encodings['input_ids']

    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader


def train_model(model, train_loader, val_loader, device, tokenizer, epochs=5, learning_rate=1e-5, output_dir="models/bart_finetuned_model"):
    """
    Train the model. Return and save the model.
    """
    # Set model to training mode
    model.train()

    # Prepare the optimizer and scheduler
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
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
            logits = outputs.logits
            probs = torch.nn.functional.gelu(logits)
            loss1 = criterion(probs.view(-1, probs.size(-1)), labels.view(-1))
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
                logits = outputs.logits
                probs = torch.nn.functional.gelu(logits)
                loss1 = criterion(probs.view(-1, probs.size(-1)), labels.view(-1))
                loss = outputs.loss
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        print(f"Validation loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the model
            model.save_pretrained(output_dir)

    return model


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
    processed_ids = []

    with torch.no_grad():
        try:
            # Iterate over test data batches
            for i, batch in enumerate(tqdm(test_data, desc="Testing")):
                input_ids, attention_mask = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                # Generate paraphrases
                outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=256, num_beams=5, early_stopping=True)

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


def evaluate_model(model, test_data, device, tokenizer):
    """
    You can use your train/validation set to evaluate models performance with the BLEU score.
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

    # subsetting the train data
    train_dataset = train_dataset.sample(frac=0.005, random_state=42)
    dev_dataset = dev_dataset.sample(frac=0.005, random_state=42)
    test_dataset =test_dataset.sample(frac=0.005, random_state=42)


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
