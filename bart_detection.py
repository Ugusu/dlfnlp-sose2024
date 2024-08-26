import argparse
import random

import numpy as np
import optimizer
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartModel
from optimizer import SophiaG, AdamW
from sklearn.metrics import matthews_corrcoef
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

TQDM_DISABLE = False


class BartWithClassifier(nn.Module):
    def __init__(self, num_labels: int = 7):
        super(BartWithClassifier, self).__init__()

        self.bart = BartModel.from_pretrained("facebook/bart-large")
        self.classifier = nn.Linear(self.bart.config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None
                ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids (torch.Tensor): Input IDs for the BART model.
            attention_mask (torch.Tensor, optional): Attention mask for the BART model.

        Returns:
            torch.Tensor: Sigmoid probabilities of the classifier output.
        """
        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]

        logits = self.classifier(cls_output)
        probabilities = self.sigmoid(logits)
        return probabilities


class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.loss = float('inf')

    def early_stop(self, val_loss):
        if val_loss < self.loss:
            self.loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def l1_loss_fn(loss_fn, model, lambda_1=0.01):
    l1_reg = sum(p.abs().sum() for p in model.parameters())
    total_loss = loss_fn + lambda_1 * l1_reg
    return total_loss


def transform_data(dataset: pd.DataFrame,
                   max_length: int = 256,
                   tokenizer_name: str = 'facebook/bart-large',
                   labels: bool = True,
                   batch_size: int = 16
                   ) -> DataLoader:
    """
    Binarizes labels ( Currently, the labels are in the form of [2, 5, 6, 0, 0, 0, 0]. This means that
    the sentence pair is of type 2, 5, and 6. Turn this into a binary form, where the
    label becomes [0, 1, 0, 0, 1, 1, 0]), tokenizes text input and transforms Dataset into torch.utils.data.DataLoader
    Args:
        dataset (pd.DataFrame): Input dataset.
        max_length (int): Maximum length for tokenizer.
        tokenizer_name (str): Tokenizer to use.
        labels (bool): If using the test dataset, set to False as there are no labels to binarize.
        batch_size (int): Batch size.

    Returns:
        DataLoader: Transformed DataLoader.
    """
    # Use AutoTokenizer from_pretrained
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

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
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        return data_loader


def train_model(model: nn.Module,
                train_data: DataLoader,
                val_data: DataLoader,
                device: torch.device,
                scheduler: torch.optim.lr_scheduler,
                optimizer: optimizer.Optimizer,
                epochs: int = 3,
                output_dir: str = "output.pt"
                ) -> nn.Module:
    """
    Trains a BartWithClassifier model for paraphrase detection, saves the model in specified output_dir, prints
    training accuracy, training loss and validation loss for each epoch and returns the model
    Args:
        model (nn.Module): Model to be trained.
        train_data (DataLoader): Training data.
        val_data (DataLoader): Validation data.
        device (torch.device): Device to be used.
        epochs (int): Number of epochs.
        output_dir (str): Directory where the model is saved.
        scheduler (torch.optim.lr_scheduler): LR Scheduler to be used
        optimizer (optimizer.Optimizer) Optimizer to be used

    Returns:
        nn.Module: Trained model.
    """
    # Loss Function and Optimizer
    loss_fn = torch.nn.CrossEntropyLoss()

    # Set best validation loss threshold
    best_matthews = float("-inf")

    # Initialize early stopping
    early_stopper = EarlyStopping(patience=3)

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
            outputs = model.forward(input_ids=b_ids, attention_mask=b_mask)
            loss = l1_loss_fn(loss_fn(outputs, b_labels), model)
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
        scheduler.step()
        # Calculate Training loss
        train_loss = train_loss / num_batches

        # Calculate training accuracy
        train_accuracy, train_matthews = evaluate_model(model=model, test_data=train_data, device=device)
        print(f"Train Accuracy: {train_accuracy}")
        print(f"Train loss: {train_loss}")
        print(f"Train Matthews Correlation Coefficient : {train_matthews}")

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
                loss = l1_loss_fn(loss_fn(outputs, b_labels), model)
                val_loss += loss.item()

        # Calculate Validation loss and accuracy
        val_accuracy, val_matthews = evaluate_model(model=model, test_data=val_data, device=device)
        val_loss = val_loss / len(val_data)
        print(f"Validation loss: {val_loss}")
        print(f"Validation accuracy: {val_accuracy}")
        print(f"Validation Matthews Correlation Coefficient : {val_matthews}")

        # Update for best Matthews Correlation Coefficient
        if val_matthews < best_matthews:
            best_matthews = val_matthews

            # Save the model
            torch.save(model, output_dir)

        # Stop early if no improvement on validation loss
        if early_stopper.early_stop(val_loss):
            break

    return model


def test_model(model: nn.Module,
               test_data: DataLoader,
               test_ids: pd.Series,
               device: torch.device
               ) -> pd.DataFrame:
    """
    Test the model. Predict the paraphrase types for the given sentences and return the results in form of
    a Pandas dataframe with the columns 'id' and 'Predicted_Paraphrase_Types'.
    Args:
        model (nn.Module): Trained model.
        test_data (DataLoader): Test data.
        test_ids (pd.Series): Test IDs.
        device (torch.device): Device to be used.

    Returns:
        pd.DataFrame: DataFrame with predictions.
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
            paraphrase_types += predicted_labels

            # Create dataframe for output
        df = pd.DataFrame({
            'id': test_ids,
            'Predicted_Paraphrase_Types': paraphrase_types
        })

    return df


def evaluate_model(model: nn.Module,
                   test_data: DataLoader,
                   device: torch.device
                   ) -> float:
    """
    This function measures the accuracy of our model's prediction on a given train/validation set
    We measure how many of the seven paraphrase types the model has predicted correctly for each data point.
    So, if the models prediction is [1,1,0,0,1,1,0] and the true label is [0,0,0,0,1,1,0], this predicition
    has an accuracy of 5/7, i.e. 71.4% .

    Args:
        model (nn.Module): Model to be evaluated.
        test_data (DataLoader): Test data.
        device (torch.device): Device to be used.

    Returns:
        float: Accuracy of the model.
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
    matthews_coefficients = []
    for label_idx in range(true_labels_np.shape[1]):
        correct_predictions = np.sum(
            true_labels_np[:, label_idx] == predicted_labels_np[:, label_idx]
        )
        total_predictions = true_labels_np.shape[0]
        label_accuracy = correct_predictions / total_predictions
        accuracies.append(label_accuracy)

        # compute Matthwes Correlation Coefficient for each paraphrase type
        matth_coef = matthews_corrcoef(true_labels_np[:, label_idx], predicted_labels_np[:, label_idx])
        matthews_coefficients.append(matth_coef)

    # Calculate the average accuracy over all labels
    accuracy = np.mean(accuracies)
    matthews_coefficient = np.mean(matthews_coefficients)
    model.train()
    return accuracy, matthews_coefficient


def seed_everything(seed: int = 11711) -> None:
    """
    Sets seed for reproducibility.

    Args:
        seed (int): Seed value.
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
    Parses command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--optimizer", type=str, default='AdamW')
    args = parser.parse_args()
    return args


def finetune_paraphrase_detection(args: argparse.Namespace) -> None:
    """
    Finetunes the model for paraphrase detection.

    Args:
        args (argparse.Namespace): Command line arguments.
    """
    model = BartWithClassifier()
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model.to(device)
    dev_dataset = pd.read_csv("data/etpc-paraphrase-dev.csv", sep="\t")
    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv", sep="\t")
    test_dataset = pd.read_csv("data/etpc-paraphrase-detection-test-student.csv", sep="\t")

    train_data = transform_data(train_dataset, max_length=args.max_length,
                                batch_size=args.batch_size)
    val_data = transform_data(dev_dataset, max_length=args.max_length,
                              batch_size=args.batch_size)
    test_data = transform_data(test_dataset, labels=False,
                               max_length=args.max_length, batch_size=args.batch_size)

    print(f"Loaded {len(train_dataset)} training samples.")

    # implement CosineAnnealing Scheduler with warmup
    if args.optimizer == 'SophiaG':
        optimizer = SophiaG(model.parameters(), lr=args.lr)
    if args.optimizer == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=args.lr)

    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=0.05 * args.lr)

    model = train_model(model, train_data, val_data, device, epochs=args.epochs, scheduler=scheduler,
                        optimizer=optimizer)

    print("Training finished.")

    accuracy, matthews_corr = evaluate_model(model, val_data, device)
    print(f"The accuracy of the model is: {accuracy:.3f}")
    print(f"Matthews Correlation Coefficient of the model is: {matthews_corr:.3f}")

    test_ids = test_dataset["id"]
    test_results = test_model(model, test_data, test_ids, device)
    test_results.to_csv(
        "predictions/bart/etpc-paraphrase-detection-test-output.csv", index=False, sep="\t"
    )


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_detection(args)
