import argparse
import os
import random
from pprint import pformat
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from bert import BertModel
from context_bert import GlobalContextLayer, GlobalContextLayerRegularized
from datasets import (
    SentenceClassificationDataset,
    SentencePairDataset,
    load_multitask_data,
)
from evaluation import model_eval_multitask, test_model_multitask
from optimizer import AdamW, SophiaG
from utils import PoolingStrategy, OptimizerType

TQDM_DISABLE = False


# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    """
    This module should use BERT for these tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    """

    def __init__(self, config):
        super(MultitaskBERT, self).__init__()

        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert parameters.
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", local_files_only=config.local_files_only
        )
        for param in self.bert.parameters():
            if config.option == "pretrain":
                param.requires_grad = False
            elif config.option == "finetune":
                param.requires_grad = True

        # Global Context Layers
        args = get_args()
        self.encoding_global_context_layer = GlobalContextLayer(hidden_size=BERT_HIDDEN_SIZE) \
            if args.regularize_context is False \
            else GlobalContextLayerRegularized(hidden_size=BERT_HIDDEN_SIZE)

        self.pooling_global_context_layer = GlobalContextLayer(hidden_size=BERT_HIDDEN_SIZE) \
            if args.regularize_context is False \
            else GlobalContextLayerRegularized(hidden_size=BERT_HIDDEN_SIZE)

        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

        self.sentiment_classifier = nn.Linear(
            in_features=BERT_HIDDEN_SIZE,  # Mapping the 768-dimension output embedding to...
            out_features=N_SENTIMENT_CLASSES  # 5 possible sentiment classes
        )

        self.paraphrase_classifier = nn.Linear(
            in_features=BERT_HIDDEN_SIZE,
            out_features=1,
        )

        self.similarity_prediction = nn.Linear(
            in_features=BERT_HIDDEN_SIZE,
            out_features=1,
        )

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                add_extra_layer: bool = False,
                pooling_strategy: PoolingStrategy = PoolingStrategy.CLS
                ) -> torch.Tensor:
        """
        Processes input sentences and produces embeddings using the BERT model based on the selected pooling strategy.

        Args:
            input_ids (torch.Tensor): Tensor of input token IDs.
            attention_mask (torch.Tensor): Tensor of attention masks.
            pooling_strategy (PoolingStrategy): Enum indicating the pooling strategy.

        Returns:
            torch.Tensor: The pooled output tensor.
        """

        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs["last_hidden_state"]  # [batch_size, seq_len, hidden_size]

        if add_extra_layer:
            sequence_output, attention_scores = self.encoding_global_context_layer(sequence_output)

        match pooling_strategy:
            case PoolingStrategy.CLS:
                # Use CLS token embedding as aggregate of whole sequence embedding
                pooled_output = sequence_output[:, 0, :]  # [batch_size, hidden_size]

            case PoolingStrategy.AVERAGE:
                # Apply average pooling over the sequence length
                pooled_output = torch.mean(sequence_output, dim=1)  # [batch_size, hidden_size]

            case PoolingStrategy.MAX:
                # Apply max pooling over the sequence length
                pooled_output, _ = torch.max(sequence_output, dim=1)  # [batch_size, hidden_size]

            case PoolingStrategy.ATTENTION:
                # Use attention scores from the Global Context Layer for pooling
                _, attention_scores = self.pooling_global_context_layer(sequence_output)  # [batch_size, seq_len, 1]

                # Sum attention scores across the second dimension (attending to all tokens)
                attention_weights = attention_scores.sum(dim=2)

                # Apply softmax to get proper weights
                attention_weights = F.softmax(attention_weights, dim=1) # [batch_size, seq_len, 1]

                # Compute weighted sum
                pooled_output = torch.bmm(attention_weights.unsqueeze(1), sequence_output).squeeze(1) # [batch_size, hidden_size]

            case _:
                raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")

        return pooled_output

    def predict_sentiment(self,
                          input_ids: torch.Tensor,
                          attention_mask: torch.Tensor,
                          add_extra_layer: bool = False,
                          pooling_strategy: PoolingStrategy = PoolingStrategy.CLS
                          ) -> torch.Tensor:
        """
        Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        Dataset: SST

        Args:
            input_ids (torch.Tensor): Tensor of input token IDs of shape (batch_size, seq_len).
            attention_mask (torch.Tensor): Tensor of attention masks of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Logits for each sentiment class for each sentence of shape (batch_size, 5).
        """
        # Get the pooled output from the forward method (CLS token's hidden state by default)
        pooled_output: torch.Tensor = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            add_extra_layer=add_extra_layer,
            pooling_strategy=pooling_strategy
        )

        # Apply dropout
        pooled_output = self.dropout(pooled_output)

        # Compute logits for sentiment classification
        logits: torch.Tensor = self.sentiment_classifier(input=pooled_output)

        return logits

    def predict_paraphrase(self,
                           input_ids_1: torch.Tensor,
                           attention_mask_1: torch.Tensor,
                           input_ids_2: torch.Tensor,
                           attention_mask_2: torch.Tensor
                           ) -> torch.Tensor:
        """
        Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        Dataset: Quora

        Args:
            input_ids_1 (torch.Tensor): The tensor of input token IDs of the 1st sequences of shape (batch_size, seq_len).
            attention_mask_1 (torch.Tensor): The tensor of attention masks of the 1st sequences of shape (batch_size, seq_len).
            input_ids_2 (torch.Tensor):The tensor of input token IDs of the 2nd sequences of shape (batch_size, seq_len).
            attention_mask_2 (torch.Tensor): The tensor of attention masks of the 2nd sequences of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: The logit predictions if the sequence pairs are paraphrases of shape (batch_size, 1).
        """

        # Context:
        # {input_ids, attention_mask}_{1,2}[:, 0] are [CLS] tokens.
        # {input_ids, attention_mask}_{1,2}[:, -1] are [SEP] tokens.
        # Removing [CLS] token from the 2nd sequences. Concatenating sequences.
        # Final shape (batch_size, seq_1_len+seq_2_len-1).
        # [ [CLS] ...seq_1... [SEP] ...seq_2... [SEP] ].
        all_input_ids = torch.cat((input_ids_1, input_ids_2[:, 1:]), dim=1)
        all_attention_mask = torch.cat((attention_mask_1, attention_mask_2[:, 1:]), dim=1)

        embedding = self.forward(all_input_ids, all_attention_mask)
        embedding = self.dropout(embedding)

        is_paraphrase_logit: torch.Tensor = self.paraphrase_classifier(embedding)

        return is_paraphrase_logit

    def predict_similarity(self,
                           input_ids_1: torch.Tensor,
                           attention_mask_1: torch.Tensor,
                           input_ids_2: torch.Tensor,
                           attention_mask_2:
                           torch.Tensor
                           ) -> torch.Tensor:
        """
        Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Since the similarity label is a number in the interval [0, 5], your output should be normalized to the interval [0, 5];
        it will be handled as a logit by the appropriate loss function.
        Dataset: STS
    
        Args:
            input_ids_1 (torch.Tensor): The tensor of input token IDs of the 1st sequences of shape (batch_size, seq_len).
            attention_mask_1 (torch.Tensor): The tensor of attention masks of the 1st sequences of shape (batch_size, seq_len).
            input_ids_2 (torch.Tensor): The tensor of input token IDs of the 2nd sequences of shape (batch_size, seq_len).
            attention_mask_2 (torch.Tensor): The tensor of attention masks of the 2nd sequences of shape (batch_size, seq_len).
    
        Returns:
            torch.Tensor: The logit predictions of the similarity score for the sequence pairs, normalized to the interval [0, 5], of shape (batch_size, 1).
        """

        all_input_ids = torch.cat((input_ids_1, input_ids_2[:, 1:]), dim=1)
        all_attention_mask = torch.cat((attention_mask_1, attention_mask_2[:, 1:]), dim=1)

        embedding = self.forward(all_input_ids, all_attention_mask)
        embedding = self.dropout(embedding)

        similarity_logit: torch.Tensor = self.similarity_prediction(embedding)

        return similarity_logit


def save_model(model, optimizer, args, config, filepath):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    save_info = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "args": args,
        "model_config": config,
        "system_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"Saving the model to {filepath}.")


def train_multitask(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

    # Load data
    # Create the data and its corresponding datasets and dataloader:
    sst_train_data, _, quora_train_data, sts_train_data = load_multitask_data(
        args.sst_train, args.quora_train, args.sts_train, split="train", subset_size=args.subset_size
    )
    sst_dev_data, _, quora_dev_data, sts_dev_data = load_multitask_data(
        args.sst_dev, args.quora_dev, args.sts_dev, split="train", subset_size=args.subset_size
    )

    sst_train_dataloader = None
    sst_dev_dataloader = None
    quora_train_dataloader = None
    quora_dev_dataloader = None
    sts_train_dataloader = None
    sts_dev_dataloader = None

    # SST dataset
    if args.task == "sst" or args.task == "multitask":
        sst_train_data = SentenceClassificationDataset(sst_train_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_train_dataloader = DataLoader(
            sst_train_data,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=sst_train_data.collate_fn,
        )
        sst_dev_dataloader = DataLoader(
            sst_dev_data,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=sst_dev_data.collate_fn,
        )

    #   Load data for the other datasets
    # If you are doing the paraphrase type detection with the minBERT model as well, make sure
    # to transform the data labels into binaries (as required in the bart_detection.py script)

    if args.task == "qqp" or args.task == "multitask":
        # Each data point: [id, seq_1, seq_2, label]. Dataset: [token_ids_1, token_type_ids_1, attention_mask_1,
        # token_ids_2, token_type_ids_2, attention_mask_2, labels, sent_ids].
        quora_train_data = SentencePairDataset(quora_train_data, args)
        quora_dev_data = SentencePairDataset(quora_dev_data, args)

        quora_train_dataloader = DataLoader(
            quora_train_data,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=quora_train_data.collate_fn,
        )

        quora_dev_dataloader = DataLoader(
            quora_dev_data,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=quora_dev_data.collate_fn,
        )

    # STS dataset
    if args.task == "sts" or args.task == "multitask":
        sts_train_data = SentencePairDataset(sts_train_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args)

        sts_train_dataloader = DataLoader(
            sts_train_data,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=sts_train_data.collate_fn,
        )
        sts_dev_dataloader = DataLoader(
            sts_dev_data,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=sts_dev_data.collate_fn,
        )

    # Init model
    config = {
        "hidden_dropout_prob": args.hidden_dropout_prob,
        "hidden_size": BERT_HIDDEN_SIZE,
        "data_dir": ".",
        "option": args.option,
        "local_files_only": args.local_files_only,
    }

    config = SimpleNamespace(**config)

    separator = "-" * 30
    print(separator)
    print("    BERT Model Configuration")
    print(separator)
    print(pformat({k: v for k, v in vars(args).items() if "csv" not in str(v)}))
    print(separator)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr

    match args.optimizer:
        case "adamw":
            optimizer = AdamW(model.parameters(), lr=lr)
        case "sophia":
            optimizer = SophiaG(model.parameters(), lr=lr)
        case _:
            raise ValueError(f"Unsupported optimizer type: {args.optimizer_type}")

    best_dev_acc = float("-inf")

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        if args.task == "sst" or args.task == "multitask":
            # Train the model on the sst dataset.

            for batch in tqdm(
                    sst_train_dataloader, desc=f"train-{epoch + 1:02}", disable=TQDM_DISABLE
            ):
                b_ids, b_mask, b_labels = (
                    batch["token_ids"],
                    batch["attention_mask"],
                    batch["labels"],
                )

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_sentiment(b_ids, b_mask, args.context_layer, args.pooling)
                loss = F.cross_entropy(logits, b_labels.view(-1))
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

        if args.task == "sts" or args.task == "multitask":
            # Trains the model on the sts dataset

            for batch in tqdm(
                    sts_train_dataloader, desc=f"train-{epoch + 1:02}", disable=TQDM_DISABLE
            ):
                b_ids_1, b_ids_2, b_mask_1, b_mask_2, b_labels = (
                    batch["token_ids_1"],
                    batch["token_ids_2"],
                    batch["attention_mask_1"],
                    batch["attention_mask_2"],
                    batch["labels"],
                )

                b_ids_1 = b_ids_1.to(device)
                b_ids_2 = b_ids_2.to(device)
                b_mask_1 = b_mask_1.to(device)
                b_mask_2 = b_mask_2.to(device)
                b_labels = b_labels.to(device).float()  # Convert labels to Float

                optimizer.zero_grad()
                logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                normalized_logits = torch.sigmoid(logits) * 5
                loss = F.mse_loss(normalized_logits, b_labels.view(-1, 1))
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

        if args.task == "qqp" or args.task == "multitask":
            # Trains the model on the qqp dataset

            for batch in tqdm(
                    quora_train_dataloader, desc=f"train-{epoch + 1:02}", disable=TQDM_DISABLE
            ):
                b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (
                    batch["token_ids_1"],
                    batch["attention_mask_1"],
                    batch["token_ids_2"],
                    batch["attention_mask_2"],
                    batch["labels"],
                )

                b_ids_1 = b_ids_1.to(device)
                b_mask_1 = b_mask_1.to(device)
                b_ids_2 = b_ids_2.to(device)
                b_mask_2 = b_mask_2.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                bce_with_logits_loss = nn.BCEWithLogitsLoss()
                loss = bce_with_logits_loss(logits.squeeze(), b_labels.float())
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

        train_loss = train_loss / num_batches

        quora_train_acc, _, _, sst_train_acc, _, _, sts_train_corr, _, _ = (
            model_eval_multitask(
                sst_train_dataloader,
                quora_train_dataloader,
                sts_train_dataloader,
                model=model,
                device=device,
                task=args.task,
            )
        )

        quora_dev_acc, _, _, sst_dev_acc, _, _, sts_dev_corr, _, _ = (
            model_eval_multitask(
                sst_dev_dataloader,
                quora_dev_dataloader,
                sts_dev_dataloader,
                model=model,
                device=device,
                task=args.task,
            )
        )

        train_acc, dev_acc = {
            "sst": (sst_train_acc, sst_dev_acc),
            "sts": (sts_train_corr, sts_dev_corr),
            "qqp": (quora_train_acc, quora_dev_acc),
            "multitask": ((sst_train_acc + sts_train_corr + quora_train_acc) / 3,
                          (sst_dev_acc + sts_dev_corr + quora_dev_acc) / 3) if args.task == "multitask" else (
            None, None),
        }[args.task]

        print(
            f"Epoch {epoch + 1:02} ({args.task}): train loss :: {train_loss:.3f}, train :: {train_acc:.3f}, dev :: {dev_acc:.3f}"
        )

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)


def test_model(args):
    with torch.no_grad():
        device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
        saved = torch.load(args.filepath)
        config = saved["model_config"]

        model = MultitaskBERT(config)
        model.load_state_dict(saved["model"])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_accuracy, quora_accuracy, sts_corr = test_model_multitask(args, model, device)

        return sst_accuracy, quora_accuracy, sts_corr


def get_args():
    parser = argparse.ArgumentParser()

    # Training task
    parser.add_argument(
        "--task",
        type=str,
        help='choose between "sst","sts","qqp","multitask" to train for different tasks ',
        choices=("sst", "sts", "qqp", "multitask"),
        default="sst",
    )

    # Model configuration
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--option",
        type=str,
        help="pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated",
        choices=("pretrain", "finetune"),
        default="pretrain",
    )
    parser.add_argument("--use_gpu", action="store_true")

    # Add this line to include subset_size
    parser.add_argument("--subset_size", type=int, default=None,
                        help="Number of examples to load from each dataset for testing")

    # Add this line to include context_layer as a boolean flag
    parser.add_argument("--context_layer", action="store_true", help="Include context layer if this flag is set.")

    # Add this line to include regularized_context as a boolean flag
    parser.add_argument("--regularize_context", action="store_true",
                        help="Use regularized context layer variant if this flag is set.")

    # Pooling strategy
    parser.add_argument(
        "--pooling",
        type=str,
        help='Choose the pooling strategy: "cls", "average", "max", or "attention".',
        choices=[strategy.value for strategy in PoolingStrategy],
        default="cls",
    )

    # Update optimizer argument
    parser.add_argument("--optimizer", type=str, default="sophia", choices=[opt.value for opt in OptimizerType],
                        help="Optimizer to use")

    args, _ = parser.parse_known_args()

    # Convert arguments to enums when necessary
    args.pooling_strategy = PoolingStrategy(args.pooling)
    args.optimizer_type = OptimizerType(args.optimizer)

    # Dataset paths
    parser.add_argument("--sst_train", type=str, default="data/sst-sentiment-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/sst-sentiment-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/sst-sentiment-test-student.csv")

    parser.add_argument("--quora_train", type=str, default="data/quora-paraphrase-train.csv")
    parser.add_argument("--quora_dev", type=str, default="data/quora-paraphrase-dev.csv")
    parser.add_argument("--quora_test", type=str, default="data/quora-paraphrase-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-similarity-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-similarity-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-similarity-test-student.csv")

    # Output paths
    parser.add_argument(
        "--sst_dev_out",
        type=str,
        default=(
            "predictions/bert/sst-sentiment-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sst-sentiment-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--sst_test_out",
        type=str,
        default=(
            "predictions/bert/sst-sentiment-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sst-sentiment-test-output.csv"
        ),
    )

    parser.add_argument(
        "--quora_dev_out",
        type=str,
        default=(
            "predictions/bert/quora-paraphrase-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/quora-paraphrase-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--quora_test_out",
        type=str,
        default=(
            "predictions/bert/quora-paraphrase-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/quora-paraphrase-test-output.csv"
        ),
    )

    parser.add_argument(
        "--sts_dev_out",
        type=str,
        default=(
            "predictions/bert/sts-similarity-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sts-similarity-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--sts_test_out",
        type=str,
        default=(
            "predictions/bert/sts-similarity-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sts-similarity-test-output.csv"
        ),
    )

    # Hyperparameters
    parser.add_argument("--batch_size", help="sst: 64 can fit a 12GB GPU", type=int, default=64)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument(
        "--lr",
        type=float,
        help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
        default=1e-3 if args.option == "pretrain" else 1e-5,
    )
    parser.add_argument("--local_files_only", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f"models/{args.option}-{args.epochs}-{args.lr}-{args.task}.pt"  # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)
