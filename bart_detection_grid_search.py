import bart_detection
import argparse
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartModel
from optimizer import SophiaG, AdamW
from sklearn.metrics import matthews_corrcoef
import torch.optim.lr_scheduler as lr_scheduler
import itertools
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

TQDM_DISABLE = False


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


def get_best_parameters(parameter_grid):
    best_matthew = -1
    best_params = None
    dev_dataset = pd.read_csv("data/etpc-paraphrase-dev.csv", sep="\t")
    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv", sep="\t")
    for lr, optim_name, batch_size in itertools.product(parameter_grid['starting_lr'],
                                                        parameter_grid['optimizer'],
                                                        parameter_grid['batch_size']):
        train_data = bart_detection.transform_data(train_dataset, max_length=256,
                                                   batch_size=batch_size)
        val_data = bart_detection.transform_data(dev_dataset, max_length=256,
                                                 batch_size=batch_size)

        model = bart_detection.BartWithClassifier()
        device = torch.device("cuda")
        model.to(device)

        if optim_name == 'SophiaG':
            optimizer = SophiaG(model.parameters(), lr=lr)
        if optim_name == 'AdamW':
            optimizer = AdamW(model.parameters(), lr=lr)
        warmup_epochs = 3
        cosine_epochs = 10 - warmup_epochs
        warmup_scheduler = LinearLR(optimizer=optimizer, start_factor=0.01, total_iters=cosine_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=cosine_epochs, eta_min=0)
        scheduler = SequentialLR(optimizer=optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                 milestones=[warmup_epochs])

        model = bart_detection.train_model(model, train_data, val_data, device, epochs=10, scheduler=scheduler,
                            optimizer=optimizer)
        accuracy, matthews_corr = bart_detection.evaluate_model(model, val_data, device)
        print(f"Learning Rate: {lr}, Batch Size: {batch_size}, Optimizer: {optim_name}, Accuracy: {accuracy:.4f},"
              f" Matthew: {matthews_corr}")

        if matthews_corr > best_matthew:
            best_matthew = matthews_corr
            best_params = {'learning_rate': lr, 'batch_size': batch_size, 'optimizer': optim_name}
    print(f"\nBest Matthew: {best_matthew:.4f}")
    print(f"Best Hyperparameters: {best_params}")

if __name__ == "__main__":
    parameter_grid = {
        'starting_lr': [1e-3, 1e-4],
        'optimizer': ['AdamW', 'SophiaG'],
        'batch_size': [16, 32],
    }
    seed_everything()
    get_best_parameters(parameter_grid)