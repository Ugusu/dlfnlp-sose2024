import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

from multitask_classifier import *

import itertools
import json
from tqdm import tqdm

import traceback
import os

from utils import PoolingStrategy, OptimizerType


def create_run_id() -> str:
    """
    Create a unique identifier for the grid search run.

    Returns:
        str: A unique identifier composed of a timestamp and a UUID.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]  # Use first 8 characters of a UUID
    return f"{timestamp}_{unique_id}"


def ensure_directory(directory: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory (str): The path of the directory to ensure.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def run_experiment(
        run_id: str,
        pooling_strategy: PoolingStrategy,
        learning_rate: float,
        hidden_dropout_prob: float,
        batch_size: int,
        epochs: int,
        optimizer_type: OptimizerType
) -> Dict[str, Any]:
    """
    Run a single experiment with the given parameters.

    Args:
        run_id (str): The unique identifier for this run.
        pooling_strategy (PoolingStrategy): The pooling strategy to use.
        learning_rate (float): The learning rate for the experiment.
        hidden_dropout_prob (float): The hidden dropout probability.
        batch_size (int): The batch size for training.
        epochs (int): The number of epochs for training.
        optimizer_type (OptimizerType): The type of optimizer to use.

    Returns:
        Dict[str, Any]: A dictionary containing the results of the experiment.
    """
    args = get_args()
    args.pooling = pooling_strategy
    args.lr = learning_rate
    args.hidden_dropout_prob = hidden_dropout_prob
    args.batch_size = batch_size
    args.epochs = epochs
    args.optimizer_type = optimizer_type

    context_layer = args.context_layer
    regularize_context = args.regularize_context
    pooling_strategy_name = pooling_strategy.value
    optimizer_name = optimizer_type.value

    # Use run_id in the filepath
    extra_context_layer_str = "-context_layer" if context_layer else ""
    regularize_context_str = "-regularize_context" if context_layer else ""
    args.filepath = (
        f"../models/{run_id}/experiment-{pooling_strategy_name}-{learning_rate}-{hidden_dropout_prob}-{batch_size}-{epochs}-{optimizer_name}"
        f"{extra_context_layer_str}{regularize_context_str}.pt")

    # Ensure the directory for this run's models exists
    ensure_directory(os.path.dirname(args.filepath))

    try:
        seed_everything(args.seed)
        train_multitask(args)

        if not os.path.exists(args.filepath):
            raise FileNotFoundError(f"Model file not found after training: {args.filepath}")

        sst_accuracy, quora_accuracy, sts_corr = test_model(args)

        # Only delete the file after successful testing
        delete_model(args.filepath)

        return {
            "extra_context_layer": context_layer,
            "regularize_context": regularize_context,
            "pooling_strategy": pooling_strategy_name,
            "learning_rate": learning_rate,
            "hidden_dropout_prob": hidden_dropout_prob,
            "batch_size": batch_size,
            "epochs": epochs,
            "optimizer": optimizer_name,
            "sst_accuracy": sst_accuracy,
            "quora_accuracy": quora_accuracy,
            "sts_correlation": sts_corr,
            "average_performance": sum(filter(None, [sst_accuracy, quora_accuracy, sts_corr])) / sum(
                1 for metric in [sst_accuracy, quora_accuracy, sts_corr] if metric is not None),
            "status": "success"
        }
    except Exception as e:
        print(f"\nError in experiment with parameters:")
        print(f"  extra_context_layer: {context_layer}")
        print(f"  regularize_context: {regularize_context}")
        print(f"  pooling_strategy: {pooling_strategy_name}")
        print(f"  learning_rate: {learning_rate}")
        print(f"  hidden_dropout_prob: {hidden_dropout_prob}")
        print(f"  batch_size: {batch_size}")
        print(f"  epochs: {epochs}")
        print(f"  optimizer_type: {optimizer_type}")
        print(f"\nError: {str(e)}")
        print("\nFull stack trace:")
        traceback.print_exc()

        # Attempt to delete the model file if it exists
        delete_model(args.filepath)

        return {
            "extra_context_layer": context_layer,
            "regularize_context": regularize_context,
            "pooling_strategy": pooling_strategy_name,
            "learning_rate": learning_rate,
            "hidden_dropout_prob": hidden_dropout_prob,
            "batch_size": batch_size,
            "epochs": epochs,
            "optimizer": optimizer_name,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def grid_search() -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Perform a grid search over specified hyperparameters.

    Returns:
        Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]: A tuple containing a list of all experiment results
        and the best result (if any successful experiments were run).
    """
    run_id = create_run_id()
    results_dir = f"grid_search_results/{run_id}"
    ensure_directory(results_dir)

    pooling_strategies = list(PoolingStrategy)
    learning_rates = [1e-5, 5e-5]
    hidden_dropout_probs = [0.3, 0.5]
    batch_sizes = [16, 32, 64]
    epochs = [5, 10]
    optimizers = list(OptimizerType)

    all_combinations = list(
        itertools.product(pooling_strategies, learning_rates, hidden_dropout_probs, batch_sizes, epochs, optimizers))

    results = []
    best_result = None

    for combo in tqdm(all_combinations, desc="Grid Search Progress"):
        result = run_experiment(run_id, *combo)
        results.append(result)

        # Print status after each experiment
        print(f"\nExperiment completed with status: {result['status']}")
        if result['status'] == 'failed':
            print(f"Error: {result['error']}")
            print("Full traceback:")
            print(result['traceback'])

        # Update best_result if this experiment was successful and better than previous best
        if result['status'] == 'success':
            if best_result is None or result['average_performance'] > best_result['average_performance']:
                best_result = result

        # Save results and best_result after each experiment
        output = {
            "run_id": run_id,
            "all_results": results,
            "best_result": best_result
        }
        with open(f"{results_dir}/grid_search_results.json", "w") as f:
            json.dump(output, f, indent=2)

    # Print best configuration
    if best_result:
        print("\nBest configuration:")
        print(json.dumps(best_result, indent=2))
    else:
        print("\nNo successful experiments found.")

    return results, best_result


def delete_model(args_filepath: str) -> None:
    """
    Delete the model file at the specified path.

    Args:
        args_filepath (str): The path to the model file to be deleted.
    """
    if os.path.exists(args_filepath):
        os.remove(args_filepath)
        print(f"Deleted model file: {args_filepath}")


if __name__ == "__main__":
    results, best_result = grid_search()
