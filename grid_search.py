import uuid
from datetime import datetime

from multitask_classifier import get_args, seed_everything, train_multitask, test_model
import itertools
import json
from tqdm import tqdm

import traceback
import os

from utils import PoolingStrategy


def create_run_id():
    """Create a unique identifier for the grid search run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]  # Use first 8 characters of a UUID
    return f"{timestamp}_{unique_id}"


def ensure_directory(directory):
    """Ensure that a directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def run_experiment(run_id, pooling_strategy, learning_rate, hidden_dropout_prob, batch_size):
    args = get_args()
    args.pooling = pooling_strategy
    args.lr = learning_rate
    args.hidden_dropout_prob = hidden_dropout_prob
    args.batch_size = batch_size

    context_layer = args.context_layer
    regularize_context = args.regularize_context
    pooling_strategy_str = pooling_strategy.value

    # Use run_id in the filepath
    extra_context_layer_str = "-context_layer" if context_layer else ""
    regularize_context_str = "-regularize_context" if context_layer else ""
    args.filepath = (f"models/{run_id}/experiment-{pooling_strategy_str}-{learning_rate}-{hidden_dropout_prob}-{batch_size}"
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
            "sophia_optimizer": "True",
            "extra_context_layer": context_layer,
            "regularize_context": regularize_context,
            "pooling_strategy": pooling_strategy_str,
            "learning_rate": learning_rate,
            "hidden_dropout_prob": hidden_dropout_prob,
            "batch_size": batch_size,
            "sst_accuracy": sst_accuracy,
            "quora_accuracy": quora_accuracy,
            "sts_correlation": sts_corr,
            "average_performance": sum(filter(None, [sst_accuracy, quora_accuracy, sts_corr])) / sum(
                1 for metric in [sst_accuracy, quora_accuracy, sts_corr] if metric is not None),
            "status": "success"
        }
    except Exception as e:
        print(f"\nError in experiment with parameters:")
        print(f"  sopia_optiizer: True")
        print(f"  extra_context_layer: {context_layer}")
        print(f"  regularize_context: {regularize_context}")
        print(f"  pooling_strategy: {pooling_strategy_str}")
        print(f"  learning_rate: {learning_rate}")
        print(f"  hidden_dropout_prob: {hidden_dropout_prob}")
        print(f"  batch_size: {batch_size}")
        print(f"\nError: {str(e)}")
        print("\nFull stack trace:")
        traceback.print_exc()

        # Attempt to delete the model file if it exists
        delete_model(args.filepath)

        return {
            "sophia_optimizer": "True",
            "extra_context_layer": context_layer,
            "regularize_context": regularize_context,
            "pooling_strategy": pooling_strategy_str,
            "learning_rate": learning_rate,
            "hidden_dropout_prob": hidden_dropout_prob,
            "batch_size": batch_size,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def grid_search():
    run_id = create_run_id()
    results_dir = f"results/{run_id}"
    ensure_directory(results_dir)

    pooling_strategies = [PoolingStrategy.CLS, PoolingStrategy.AVERAGE, PoolingStrategy.MAX, PoolingStrategy.ATTENTION]
    learning_rates = [1e-5, 5e-5]
    hidden_dropout_probs = [0.3, 0.5]
    batch_sizes = [16, 32, 64]
        

    all_combinations = list(itertools.product(pooling_strategies, learning_rates, hidden_dropout_probs, batch_sizes))

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
    if os.path.exists(args_filepath):
        os.remove(args_filepath)
        print(f"Deleted model file: {args_filepath}")


if __name__ == "__main__":
    results, best_result = grid_search()
