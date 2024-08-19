from multitask_classifier import get_args, seed_everything, train_multitask, test_model
import itertools
import json
from tqdm import tqdm

import traceback
import os


def run_experiment(pooling_strategy, learning_rate, hidden_dropout_prob, batch_size):
    args = get_args()
    args.pooling = pooling_strategy
    args.lr = learning_rate
    args.hidden_dropout_prob = hidden_dropout_prob
    args.batch_size = batch_size
    args.filepath = f"models/experiment-{pooling_strategy}-{learning_rate}-{hidden_dropout_prob}-{batch_size}.pt"

    # Saved model for testing
    # args.filepath = f"models/experiment-cls-1e-05-0.3-64.pt"

    try:
        seed_everything(args.seed)
        train_multitask(args)

        if not os.path.exists(args.filepath):
            raise FileNotFoundError(f"Model file not found after training: {args.filepath}")

        sst_accuracy, quora_accuracy, sts_corr = test_model(args)

        # Only delete the file after successful testing
        os.remove(args.filepath)
        print(f"Deleted model file: {args.filepath}")

        return {
            "pooling_strategy": pooling_strategy,
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
        print(f"  pooling_strategy: {pooling_strategy}")
        print(f"  learning_rate: {learning_rate}")
        print(f"  hidden_dropout_prob: {hidden_dropout_prob}")
        print(f"  batch_size: {batch_size}")
        print(f"\nError: {str(e)}")
        print("\nFull stack trace:")
        traceback.print_exc()

        # Attempt to delete the model file if it exists
        if os.path.exists(args.filepath):
            os.remove(args.filepath)
            print(f"Deleted model file after error: {args.filepath}")

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
        learning_rates = [1e-5]
        hidden_dropout_probs = [0.3]
        batch_sizes = [64]
    else:
        pooling_strategies = ["cls", "average", "max", "attention"]
        learning_rates = [1e-5, 3e-5, 5e-5]
        hidden_dropout_probs = [0.1, 0.3, 0.5]
        batch_sizes = [16, 32, 64]

    all_combinations = list(itertools.product(pooling_strategies, learning_rates, hidden_dropout_probs, batch_sizes))

    results = []
    best_result = None

    for combo in tqdm(all_combinations, desc="Grid Search Progress"):
        result = run_experiment(*combo)
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
            "all_results": results,
            "best_result": best_result
        }
        with open("grid_search_results.json", "w") as f:
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
        print(f"Deleted model file after error: {args_filepath}")


if __name__ == "__main__":
    results, best_result = grid_search()
