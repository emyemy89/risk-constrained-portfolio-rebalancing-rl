"""
Debugging and analysis utilities for the experiments.

This module runs post-training diagnostics for portfolio RL models. It evaluates
trained agents, compares performance against baseline strategies, analyzes
portfolio allocations, computes statistics, and generates plots for experiment
results.
"""
import os
import io
import contextlib
import numpy as np
import matplotlib.pyplot as plt

from train.evaluate import (inspect_weights, evaluate_model, compute_metrics,
                            evaluate_baseline, weight_statistics, turnover_statistics,
                            equal_weight_distance, cash_statistics, regime_analysis)
from train.utils.inspect_data import plot_portfolios, plot_weights, plot_cash_weight
from train.experiments.momentum_baseline import run_momentum_strategy, calculate_metrics


def run_debugging_info(model, test_env, test_returns, seed, fold_idx):
    """
    Run post-training evaluation and save experiment diagnostics.

    Evaluates the trained RL model on the test environment, compares it against
    baseline strategies, prints portfolio statistics, saves debug logs, and
    generates evaluation plots.
    """
    out_dir = f"./results/fold_{fold_idx}/seed_{seed}"
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "debug_info.txt")

    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        print(f"=== Debug info for seed {seed} ===")

        inspect_weights(model, test_env)

        results = evaluate_model(model, test_env)

        print("Final portfolio value:", results["portfolio_values"][-1])
        total_return = results["portfolio_values"][-1] - 1
        print("Test cumulative return:", total_return, "\n")

        metrics = compute_metrics(results["portfolio_values"], results["daily_returns"])
        momentum_returns = run_momentum_strategy(test_returns, lookback=20)

        spy = evaluate_baseline(test_returns, np.array([1, 0, 0, 0, 0, 0]))
        equal = evaluate_baseline(test_returns, np.ones(6) / 6)
        print("--- Baseline comparison ---")
        print(f"{rl_algorithm}:", metrics)
        print("SPY:", spy)
        print("Equal:", equal, "\n")

        print("--- Weight statistics ---")
        weight_statistics(results["weights"])

        print(f"\n--- {rl_algorithm} metrics ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print()

        print("\n--- Momentum ---")
        print(calculate_metrics(momentum_returns))

        print("\n--- Turnover statistics ---")
        turnover = turnover_statistics(results["weights"])
        # this one returns an array -> summarize instead of dumping raw numbers
        print(f"mean: {np.mean(turnover):.4f}  std: {np.std(turnover):.4f}  "
              f"min: {np.min(turnover):.4f}  max: {np.max(turnover):.4f}")

        print("\n--- Equal weight distance ---")
        equal_weight_distance(results["weights"])

        print("\n--- Cash statistics ---")
        cash_statistics(results["weights"], test_returns)

        print("\n--- Regime analysis ---")
        regime_analysis(results["weights"], test_returns)

    output = buffer.getvalue()
    print(output)  # still show it in console

    with open(log_path, "w") as f:
        f.write(output)
    print(f"\nSaved debug info to {log_path}")

    # plots
    rl_algorithm_values = results["portfolio_values"]
    spy_returns = test_returns @ np.array([1, 0, 0, 0, 0, 0])
    equal_returns = test_returns @ (np.ones(6) / 6)
    spy_values = np.exp(np.cumsum(spy_returns))
    equal_values = np.exp(np.cumsum(equal_returns))

    plot_portfolios({f"{rl_algorithm}": rl_algorithm_values, "SPY": spy_values, "Equal Weight": equal_values})
    plt.savefig(os.path.join(out_dir, "portfolio_values.png"))
    plt.close()

    plot_weights(results["weights"])
    plt.savefig(os.path.join(out_dir, "weights.png"))
    plt.close()

    plot_cash_weight(results["weights"])
    plt.savefig(os.path.join(out_dir, "cash_weight.png"))
    plt.close()