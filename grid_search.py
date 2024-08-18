import sys

from multitask_classifier import get_args, seed_everything, train_multitask, test_model
import itertools
import json
from tqdm import tqdm

import traceback
import os


def run_experiment(pooling_strategy, learning_rate, hidden_dropout_prob, batch_size):
    try:
        args = get_args()
        args.pooling = pooling_strategy
        args.lr = learning_rate
        args.hidden_dropout_prob = hidden_dropout_prob
        args.batch_size = batch_size
        args.filepath = f"models/experiment-{pooling_strategy}-{learning_rate}-{hidden_dropout_prob}-{batch_size}.pt"

        seed_everything(args.seed)
        train_multitask(args)
        result = test_model(args)

        if result is None:
            raise ValueError("test_model returned None")

        quora_accuracy, _, _, sst_accuracy, _, _, sts_corr, _, _ = result

        # Delete the saved model file after evaluation
        delete_model(args.filepath)

        return {
            "pooling_strategy": pooling_strategy,
            "learning_rate": learning_rate,
            "hidden_dropout_prob": hidden_dropout_prob,
            "batch_size": batch_size,
            "quora_accuracy": quora_accuracy,
            "sst_accuracy": sst_accuracy,
            "sts_correlation": sts_corr,
            "average_performance": (quora_accuracy + sst_accuracy + sts_corr) / 3,
            "status": "success"
        }
    except Exception as e:
        print(f"\nError in experiment with parameters:")
        print(f"  pooling_strategy: {pooling_strategy}")
        print(f"  learning_rate: {learning_rate}")
        print(f"  hidden_dropout_prob: {hidden_dropout_prob}")
        print(f"  batch_size: {batch_size}")
        print("\nFull stack trace:")
        traceback.print_exc(file=sys.stdout)

        return {
            "pooling_strategy": pooling_strategy,
            "learning_rate": learning_rate,
            "hidden_dropout_prob": hidden_dropout_prob,
            "batch_size": batch_size,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def grid_search():
    
    run_test = True

    if run_test:
        pooling_strategies = ["cls", "average"]
        learning_rates = [1e-5, 1e-5]
        hidden_dropout_probs = [0.3]
        batch_sizes = [64]
    else:
        pooling_strategies = ["cls", "average", "max", "attention"]
        learning_rates = [1e-5, 3e-5, 5e-5]
        hidden_dropout_probs = [0.1, 0.3, 0.5]
        batch_sizes = [16, 32, 64]

    all_combinations = list(itertools.product(pooling_strategies, learning_rates, hidden_dropout_probs, batch_sizes))

    results = []

    for combo in tqdm(all_combinations, desc="Grid Search Progress"):
        result = run_experiment(*combo)
        results.append(result)

        # Print status after each experiment
        print(f"\nExperiment completed with status: {result['status']}")
        if result['status'] == 'failed':
            print(f"Error: {result['error']}")
            print("Full traceback:")
            print(result['traceback'])

        # Save results after each experiment
        with open("grid_search_results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Find best configuration
    successful_results = [r for r in results if r['status'] == 'success']
    if successful_results:
        best_result = max(successful_results, key=lambda x: x["average_performance"])
        print("\nBest configuration:")
        print(json.dumps(best_result, indent=2))
    else:
        print("\nNo successful experiments found.")
        best_result = None

    return results, best_result


def delete_model(args_filepath: str) -> None:
    if os.path.exists(args_filepath):
        os.remove(args_filepath)
        print(f"Deleted model file after error: {args_filepath}")


if __name__ == "__main__":
    results, best_result = grid_search()
