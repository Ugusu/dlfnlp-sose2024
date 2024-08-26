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


from bert_mean_pooling import BertModel

from datasets import (
    SentenceClassificationDataset,
    SentencePairDataset,
    load_multitask_data,
)
from evaluation import model_eval_multitask, test_model_multitask
from optimizer import AdamW

from transformers import get_linear_schedule_with_warmup

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
    

    def __init__(self, config):
        super(MultitaskBERT, self).__init__()

        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert parameters.
        self.bert = BertModel.from_pretrained(
            "google-bert/bert-base-uncased", local_files_only=config.local_files_only
        )
        for param in self.bert.parameters():
            if config.option == "pretrain":
                param.requires_grad = False
            elif config.option == "finetune":
                param.requires_grad = True

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
                return_pooler_output: bool = True
                ) -> torch.Tensor:
        

        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).

        outputs = self.bert(input_ids, attention_mask=attention_mask)

        if return_pooler_output:
            return outputs["pooler_output"]  # CLS token output
        else:
            return outputs["last_hidden_state"]  # Sequence of hidden states

    def predict_sentiment(self,
                          input_ids: torch.Tensor,
                          attention_mask: torch.Tensor
                          ) -> torch.Tensor:
        
        # Get the pooled output from the forward method (CLS token's hidden state by default)
        pooled_output: torch.Tensor = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask
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
    
    
        embeddings_a = model.forward(b_ids_1, b_mask_1)
        embeddings_b = model.forward(b_ids_2, b_mask_2)
        embeddings_a = model.dropout(embeddings_a)
        embeddings_b = model.dropout(embeddings_b)
                
        cosine_sim: torch.Tensor = F.cosine_similarity(embeddings_a, embeddings_b)
        
        return  2.5 + 2.5 * F.cosine_similarity(embeddings_a, embeddings_b)



    def predict_paraphrase_types(
            self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
    ):
        
        ### TODO
        raise NotImplementedError


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
    sst_train_data, _, quora_train_data, sts_train_data, etpc_train_data = load_multitask_data(
        args.sst_train, args.quora_train, args.sts_train, args.etpc_train, split="train", subset_size=args.subset_size
    )
    sst_dev_data, _, quora_dev_data, sts_dev_data, etpc_dev_data = load_multitask_data(
        args.sst_dev, args.quora_dev, args.sts_dev, args.etpc_dev, split="train", subset_size=args.subset_size
    )

    sst_train_dataloader = None
    sst_dev_dataloader = None
    quora_train_dataloader = None
    quora_dev_dataloader = None
    sts_train_dataloader = None
    sts_dev_dataloader = None
    etpc_train_dataloader = None
    etpc_dev_dataloader = None

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
  
    
    
    #Create checkpoint of last model
    filepath = f"models/{args.option}-{args.epochs}-{args.lr}-qqp.pt"
    checkpoint = torch.load(filepath)
    
    #Define model and load last state
    model = MultitaskBERT(config)
    model.load_state_dict(checkpoint['model'])
    
    
    model = model.to(device)
   
    #Define optimizer and load last state
    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    optimizer.load_state_dict(checkpoint['optim'])
    
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
                b_labels = b_labels.to(device).float()

                optimizer.zero_grad()
                logits = model.predict_sentiment(b_ids, b_mask)
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
                b_labels = b_labels.to(device).float()

                optimizer.zero_grad()
                normalized_logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
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
                b_labels = b_labels.to(device).float()

                optimizer.zero_grad()
                logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                bce_with_logits_loss = nn.BCEWithLogitsLoss()
                loss = bce_with_logits_loss(logits.squeeze(), b_labels.float())
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

        if args.task == "etpc" or args.task == "multitask":
            # Trains the model on the etpc dataset
            pass
            # raise NotImplementedError

        train_loss = train_loss / num_batches

        quora_train_acc, _, _, sst_train_acc, _, _, sts_train_corr, _, _, etpc_train_acc, _, _ = (
            model_eval_multitask(
                sst_train_dataloader,
                quora_train_dataloader,
                sts_train_dataloader,
                etpc_train_dataloader,
                model=model,
                device=device,
                task=args.task,
            )
        )

        quora_dev_acc, _, _, sst_dev_acc, _, _, sts_dev_corr, _, _, etpc_dev_acc, _, _ = (
            model_eval_multitask(
                sst_dev_dataloader,
                quora_dev_dataloader,
                sts_dev_dataloader,
                etpc_dev_dataloader,
                model=model,
                device=device,
                task=args.task,
            )
        )

        train_acc, dev_acc = {
            "sst": (sst_train_acc, sst_dev_acc),
            "sts": (sts_train_corr, sts_dev_corr),
            "qqp": (quora_train_acc, quora_dev_acc),
            "etpc": (etpc_train_acc, etpc_dev_acc),
            "multitask": (0, 0),  # TODO
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

        return test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()

    # Training task
    parser.add_argument(
        "--task",
        type=str,
        help='choose between "sst","sts","qqp","etpc","multitask" to train for different tasks ',
        choices=("sst", "sts", "qqp", "etpc", "multitask"),
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

    args, _ = parser.parse_known_args()

    # Dataset paths
    parser.add_argument("--sst_train", type=str, default="/kaggle/input/dlfnlp/dlfnlp-sose2024/data/sst-sentiment-train.csv")
    parser.add_argument("--sst_dev", type=str, default="/kaggle/input/dlfnlp/dlfnlp-sose2024/data/sst-sentiment-dev.csv")
    parser.add_argument("--sst_test", type=str, default="/kaggle/input/dlfnlp/dlfnlp-sose2024/data/sst-sentiment-test-student.csv")

    parser.add_argument("--quora_train", type=str, default="/kaggle/input/dlfnlp/dlfnlp-sose2024/data/quora-paraphrase-train.csv")
    parser.add_argument("--quora_dev", type=str, default="/kaggle/input/dlfnlp/dlfnlp-sose2024/data/quora-paraphrase-dev.csv")
    parser.add_argument("--quora_test", type=str, default="/kaggle/input/dlfnlp/dlfnlp-sose2024/data/quora-paraphrase-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="/kaggle/input/dlfnlp/dlfnlp-sose2024/data/sts-similarity-train.csv")
    parser.add_argument("--sts_dev", type=str, default="/kaggle/input/dlfnlp/dlfnlp-sose2024/data/sts-similarity-dev.csv")
    parser.add_argument("--sts_test", type=str, default="/kaggle/input/dlfnlp/dlfnlp-sose2024/data/sts-similarity-test-student.csv")

    parser.add_argument("--etpc_train", type=str, default="/kaggle/input/dlfnlp/dlfnlp-sose2024/data/etpc-paraphrase-train.csv")
    parser.add_argument("--etpc_dev", type=str, default="/kaggle/input/dlfnlp/dlfnlp-sose2024/data/etpc-paraphrase-dev.csv")
    parser.add_argument("--etpc_test", type=str, default="/kaggle/input/dlfnlp/dlfnlp-sose2024/data/etpc-paraphrase-detection-test-student.csv")

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
            "predictions/bert/sts-main-similarity-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sts-similarity-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--sts_test_out",
        type=str,
        default=(
            "predictions/bert/sts-main-similarity-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sts-similarity-test-output.csv"
        ),
    )

    parser.add_argument(
        "--etpc_dev_out",
        type=str,
        default=(
            "predictions/bert/etpc-paraphrase-detection-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/etpc-paraphrase-detection-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--etpc_test_out",
        type=str,
        default=(
            "predictions/bert/etpc-paraphrase-detection-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/etpc-paraphrase-detection-test-output.csv"
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
