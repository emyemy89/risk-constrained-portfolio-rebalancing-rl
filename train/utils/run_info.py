import os
import io
import contextlib
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from train.evaluate import *
from train.utils.inspect_data import plot_portfolios, plot_weights, plot_cash_weight


def run_debugging_info(model, test_env, test_returns, seed):
    out_dir = f"./results/seed_{seed}"
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "debug_info.txt")

    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        print(f"=== Debug info for seed {seed} ===")

        inspect_weights(model, test_env)

        best_model = PPO.load(f"../models/best_model_seed_{seed}/best_model", env=test_env)
        results = evaluate_portfolio(best_model, test_env)

        print("Final portfolio value:", results["portfolio_values"][-1])
        total_return = results["portfolio_values"][-1] - 1
        print("Test cumulative return:", total_return, "\n")

        metrics = compute_metrics(results["portfolio_values"], results["daily_returns"])

        spy = evaluate_baseline(test_returns, np.array([1, 0, 0, 0, 0, 0]))
        equal = evaluate_baseline(test_returns, np.ones(6) / 6)
        print("--- Baseline comparison ---")
        print("PPO:", metrics)
        print("SPY:", spy)
        print("Equal:", equal, "\n")

        print("--- PPO metrics ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print()

        print("--- Weight statistics ---")
        weight_statistics(results["weights"])

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
    ppo_values = results["portfolio_values"]
    spy_returns = test_returns @ np.array([1, 0, 0, 0, 0, 0])
    equal_returns = test_returns @ (np.ones(6) / 6)
    spy_values = np.exp(np.cumsum(spy_returns))
    equal_values = np.exp(np.cumsum(equal_returns))

    plot_portfolios({"PPO": ppo_values, "SPY": spy_values, "Equal Weight": equal_values})
    plt.savefig(os.path.join(out_dir, "portfolio_values.png"))

    plot_weights(results["weights"])
    plt.savefig(os.path.join(out_dir, "weights.png"))

    plot_cash_weight(results["weights"])
    plt.savefig(os.path.join(out_dir, "cash_weight.png"))