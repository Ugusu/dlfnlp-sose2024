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
        quora_accuracy, _, _, sst_accuracy, _, _, sts_corr, _, _ = test_model(args)

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
        print(f"Error in experiment: {str(e)}")
        traceback.print_exc()

        # Attempt to delete the model file even if an error occurred
        delete_model(args.filepath)

        return {
            "pooling_strategy": pooling_strategy,
            "learning_rate": learning_rate,
            "hidden_dropout_prob": hidden_dropout_prob,
            "batch_size": batch_size,
            "status": "failed",
            "error": str(e)
        }


def grid_search():
    pooling_strategies = ["cls", "average", "max", "attention"]
    # learning_rates = [1e-5, 3e-5, 5e-5]
    # hidden_dropout_probs = [0.1, 0.3, 0.5]
    # batch_sizes = [16, 32, 64]

    learning_rates = [1e-5, 1e-5]
    hidden_dropout_probs = [0.3]
    batch_sizes = [64]

    all_combinations = list(itertools.product(pooling_strategies, learning_rates, hidden_dropout_probs, batch_sizes))

    results = []

    for combo in tqdm(all_combinations, desc="Grid Search Progress"):
        result = run_experiment(*combo)
        results.append(result)

        # Save results after each experiment
        with open("grid_search_results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Find best configuration
    best_result = max(results, key=lambda x: x["average_performance"])

    print("Best configuration:")
    print(json.dumps(best_result, indent=2))

    return results, best_result


def delete_model(args_filepath: str) -> None:
    if os.path.exists(args_filepath):
        os.remove(args_filepath)
        print(f"Deleted model file after error: {args_filepath}")


if __name__ == "__main__":
    results, best_result = grid_search()
