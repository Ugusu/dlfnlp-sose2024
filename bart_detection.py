import argparse
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartModel
from optimizer import AdamW
import torch.nn.functional as F

TQDM_DISABLE = False


class BartWithClassifier(nn.Module):
    def __init__(self, config):
        super(BartWithClassifier, self).__init__()

        self.bart = BartModel.from_pretrained("facebook/bart-large", local_files_only=config.local_files_only)
        self.classifier = nn.Linear(self.bart.config.hidden_size, config.num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None):
        # Use the BartModel to obtain the last hidden state
        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]

        # Add an additional fully connected layer to obtain the logits
        logits = self.classifier(cls_output)

        # Return the probabilities
        probabilities = self.sigmoid(logits)
        return probabilities


def transform_data(dataset, max_length=256, tokenizer_name='facebook/bart-large', labels=True, batch_size=16,
                   local_files_only=False):
    """
    Binarizes labels ( Currently, the labels are in the form of [2, 5, 6, 0, 0, 0, 0]. This means that
    the sentence pair is of type 2, 5, and 6. Turn this into a binary form, where the
    label becomes [0, 1, 0, 0, 1, 1, 0]), tokenizes text input and transforms Dataset into torch.utils.data.DataLoader
    Args
        dataset (pd.DataFrame): input dataset
        max_length (int): maximum length for tokenizer
        tokenizer_name (str): tokenizer to use
        labels (bool): If using the test dataset, set to false, as there are no labels to binarize
        batch_size (int): batch size
        local_files_only (bool): set to true to only use local files

    Returns:
        data_loader (torch.utils.data.DataLoader): transformed DataLoader
    Turn the data to the format you want to use.
    """
    # Use AutoTokenizer from_pretrained
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=local_files_only)

    # Binarize labels if there are labels
    if labels:
        binary_labels = []
        for row in dataset['paraphrase_types']:
            labels = np.zeros(7, dtype=float)
            for i in range(1, 8):
                if str(i) in row:
                    labels[i - 1] = 1
            binary_labels.append(labels)

        # Tokenize data
        encodings = tokenizer((dataset['sentence1'].to_list()), (dataset['sentence2'].to_list()), truncation=True,
                              padding=True, max_length=max_length, return_tensors='pt')

        # Create dataset
        dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'],
                                torch.tensor(np.array(binary_labels)))
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return data_loader

    else:
        # Tokenize data
        encodings = tokenizer((dataset['sentence1'].to_list()), (dataset['sentence2'].to_list()), truncation=True,
                              padding=True, max_length=max_length, return_tensors='pt')
        # Create dataset
        dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'])
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return data_loader


def train_model(model, train_data, val_data, device, learning_rate=1e-5, epochs=3, output_dir="output.pt"):
    """
    Trains a BartWithClassifier model for paraphrase detection, saves the model in specified output_dir, prints
    training accuracy, training loss and validation loss for each epoch and returns the model
    Args:
        model: model to be trained
        train_data (torch.utils.data.DataLoader): training data transformed using transform_data function
        val_data (torch.utils.data.DataLoader): validation data transformed using transform_data function
        device (torch.device): device to be used
        learning_rate (float): learning rate to be used
        epochs (int): number of epochs
        output_dir (str): directory where model is saved

    Returns:
        model: trained model
    """
    # Loss Function and Optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Set best validation loss threshold
    best_val_loss = float("inf")

    # Loop over epochs
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = 0
        num_batches = 0

        # Set to training mode
        model.train()

        # Train Model
        for batch in tqdm(train_data, desc=f"train-{epoch + 1:02}"):
            b_ids, b_mask, b_labels = (
                batch[0],
                batch[1],
                batch[2],
            )
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=b_ids, attention_mask=b_mask)
            loss = loss_fn(outputs, b_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        # Calculate Training loss
        train_loss = train_loss / num_batches

        # Calculate training accuracy
        train_accuracy = evaluate_model(model=model, test_data=train_data, device=device)
        print(f"Train Accuracy: {train_accuracy}")
        print(f"Train loss: {train_loss}")

        val_loss = 0
        model.eval()

        # Evaluate on Validation set
        with torch.no_grad():
            for batch in tqdm(val_data, desc=f"Validation"):
                b_ids, b_mask, b_labels = (
                    batch[0],
                    batch[1],
                    batch[2],
                )
                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_labels = b_labels.to(device)

                outputs = model(input_ids=b_ids, attention_mask=b_mask)
                loss = loss_fn(outputs, b_labels)
                val_loss += loss.item()

        # Calculate Validation loss and accuracy
        val_accuracy = evaluate_model(model=model, test_data=val_data, device=device)
        val_loss = val_loss / len(val_data)
        print(f"Validation loss: {val_loss}")
        print(f"Validation accuracy: {val_accuracy}")

        # Update for best Validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            # Save the model
            torch.save(model, output_dir)

    return model


def test_model(model, test_data, test_ids, device):
    """
     Test the model. Predict the paraphrase types for the given sentences and return the results in form of
    a Pandas dataframe with the columns 'id' and 'Predicted_Paraphrase_Types'.
    Args:
        model: trained model
        test_data (torch.utils.data.DataLoader): test data transformed using transform_data function
        test_ids (pd.Series): test ids of the test dataset
        device (torch.device): device to be used

    Returns:
        df (pd.DataFrame):  a Pandas dataframe with the columns 'id' and 'Predicted_Paraphrase_Types'.
    """
    # Set model to evaluation mode
    model.eval()

    paraphrase_types = []

    # Test the model
    with torch.no_grad():
        for batch in tqdm(test_data, desc=f"Testing"):
            b_ids, b_mask = (
                batch[0],
                batch[1],
            )

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            outputs = model(input_ids=b_ids, attention_mask=b_mask)

            # Set labels
            predicted_labels = (outputs > 0.5).int()

            predicted_labels = predicted_labels.tolist()
            print(predicted_labels)
            paraphrase_types += predicted_labels

            # Create dataframe for ouput
        df = pd.DataFrame({
            'id': test_ids,
            'Predicted_Paraphrase_Types': paraphrase_types
        })

    return df


def evaluate_model(model, test_data, device):
    """
    This function measures the accuracy of our model's prediction on a given train/validation set
    We measure how many of the seven paraphrase types the model has predicted correctly for each data point.
    So, if the models prediction is [1,1,0,0,1,1,0] and the true label is [0,0,0,0,1,1,0], this predicition
    has an accuracy of 5/7, i.e. 71.4% .
    """
    all_pred = []
    all_labels = []
    model.eval()

    with torch.no_grad():
        for batch in test_data:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predicted_labels = (outputs > 0.5).int()

            all_pred.append(predicted_labels)
            all_labels.append(labels)

    all_predictions = torch.cat(all_pred, dim=0)
    all_true_labels = torch.cat(all_labels, dim=0)

    true_labels_np = all_true_labels.cpu().numpy()
    predicted_labels_np = all_predictions.cpu().numpy()

    # Compute the accuracy for each label
    accuracies = []
    for label_idx in range(true_labels_np.shape[1]):
        correct_predictions = np.sum(
            true_labels_np[:, label_idx] == predicted_labels_np[:, label_idx]
        )
        total_predictions = true_labels_np.shape[0]
        label_accuracy = correct_predictions / total_predictions
        accuracies.append(label_accuracy)

    # Calculate the average accuracy over all labels
    accuracy = np.mean(accuracies)
    model.train()
    return accuracy


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
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num_labels", type=int, default=7)
    args = parser.parse_args()
    return args


def finetune_paraphrase_detection(args):
    model = BartWithClassifier(config=args)
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model.to(device)
    dev_dataset = pd.read_csv("data/etpc-paraphrase-dev.csv", sep="\t")
    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv", sep="\t")
    test_dataset = pd.read_csv("data/etpc-paraphrase-detection-test-student.csv", sep="\t")

    train_data = transform_data(train_dataset, local_files_only=args.local_files_only, max_length=args.max_length,
                                batch_size=args.batch_size)
    val_data = transform_data(dev_dataset, local_files_only=args.local_files_only, max_length=args.max_length,
                              batch_size=args.batch_size)
    test_data = transform_data(test_dataset, labels=False, local_files_only=args.local_files_only,
                               max_length=args.max_length, batch_size=args.batch_size)

    print(f"Loaded {len(train_dataset)} training samples.")

    model = train_model(model, train_data, val_data, device, learning_rate=args.lr, epochs=args.epochs)

    print("Training finished.")

    accuracy = evaluate_model(model, val_data, device)
    print(f"The accuracy of the model is: {accuracy:.3f}")

    test_ids = test_dataset["id"]
    test_results = test_model(model, test_data, test_ids, device)
    test_results.to_csv(
        "predictions/bart/etpc-paraphrase-detection-test-output.csv", index=False, sep="\t"
    )


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_detection(args)
