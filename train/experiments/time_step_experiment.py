import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from data.pipeline import load_data
from train.make_env import make_env
from train.utils.algorithm_selection import create_model
from train.evaluate import evaluate_and_compute_metrics


SEEDS = [0, 1, 2, 3, 4]

FOLDS = [
    ("2012-12-31", "2013-01-01", "2014-12-31"),
    ("2014-12-31", "2015-01-01", "2016-12-31"),
    ("2016-12-31", "2017-01-01", "2018-12-31"),
    ("2018-12-31", "2019-01-01", "2020-12-31"),
]

# Training budgets to compare
TIMESTEPS_LIST = [10_000, 50_000, 100_000, 200_000, 400_000, 800_000, 1_000_000]


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_timestep_experiment(rl_algorithm="PPO"):
    """
    Evaluate how training duration affects RL portfolio performance.

    For every timestep budget, train one model for each fold and seed,
    evaluate it on the corresponding validation period, and store the
    resulting performance metrics.
    """
    all_results = []
    for total_timesteps in TIMESTEPS_LIST:
        print("\n" + "=" * 70)
        print(f"Testing total_timesteps = {total_timesteps:,}")
        print("=" * 70)
        for fold_idx, (train_end, val_start, val_end) in enumerate(FOLDS):
            print(
                f"\nFold {fold_idx + 1}/{len(FOLDS)} "
                f"({train_end} -> {val_start} -> {val_end})"
            )
            # Load data once for this fold.
            (
                train_windows,
                train_returns,
                val_windows,
                val_returns,
                feature_columns,
            ) = load_data(
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
            )
            for seed in SEEDS:
                print(
                    f"  timesteps={total_timesteps:,}, "
                    f"fold={fold_idx + 1}, seed={seed}"
                )
                set_seed(seed)
                # Create fresh environments for every experiment.
                train_env = make_env(train_windows, train_returns,)
                val_env = make_env(val_windows, val_returns,)

                train_env.reset(seed=seed)
                val_env.reset(seed=seed)

                # Create a fresh model.
                model = create_model(rl_algorithm, train_env, seed,)
                # Train for the current timestep budget.
                model.learn(total_timesteps=total_timesteps)
                # Evaluate on validation data.
                metrics = evaluate_and_compute_metrics(model, val_env,)

                result = {
                    "timesteps": total_timesteps,
                    "fold": fold_idx + 1,
                    "seed": seed,
                    **metrics,
                }

                all_results.append(result)

                # Explicitly close environments.
                train_env.close()
                val_env.close()

    results_df = pd.DataFrame(all_results)

    # Save raw results.
    results_df.to_csv("timestep_experiment_results.csv", index=False,)

    print("\n=== Raw Results ===")
    print(results_df)

    # Aggregate over folds and seeds.
    summary = (
        results_df
        .groupby("timesteps")
        .agg(
            mean_return=("Annual Return", "mean"),
            std_return=("Annual Return", "std"),

            mean_volatility=("Annual Volatility", "mean"),
            std_volatility=("Annual Volatility", "std"),

            mean_sharpe=("Sharpe Ratio", "mean"),
            std_sharpe=("Sharpe Ratio", "std"),

            mean_drawdown=("Max Drawdown", "mean"),
            std_drawdown=("Max Drawdown", "std"),
        )
        .reset_index()
    )

    print("\n=== Timestep Summary ===")
    print(summary)

    summary.to_csv(
        "timestep_experiment_summary.csv",
        index=False,
    )

    plot_results(summary)

    return results_df, summary


def plot_results(summary):
    """Plot validation performance against training timesteps."""

    timesteps = summary["timesteps"]

    # ---------------------------------------------------------
    # Annual Return
    # ---------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.errorbar(
        timesteps,
        summary["mean_return"],
        yerr=summary["std_return"],
        marker="o",
        capsize=4,
    )

    plt.xscale("log")

    plt.xlabel("Training Timesteps")
    plt.ylabel("Annual Return")
    plt.title("Annual Return vs Training Timesteps")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "timestep_annual_return.png",
        dpi=300,
    )

    plt.show()

    # ---------------------------------------------------------
    # Sharpe Ratio
    # ---------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.errorbar(
        timesteps,
        summary["mean_sharpe"],
        yerr=summary["std_sharpe"],
        marker="o",
        capsize=4,
    )

    plt.xscale("log")

    plt.xlabel("Training Timesteps")
    plt.ylabel("Sharpe Ratio")
    plt.title("Sharpe Ratio vs Training Timesteps")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "timestep_sharpe.png",
        dpi=300,
    )

    plt.show()

    # ---------------------------------------------------------
    # Volatility
    # ---------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.errorbar(
        timesteps,
        summary["mean_volatility"],
        yerr=summary["std_volatility"],
        marker="o",
        capsize=4,
    )

    plt.xscale("log")

    plt.xlabel("Training Timesteps")
    plt.ylabel("Annual Volatility")
    plt.title("Annual Volatility vs Training Timesteps")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "timestep_volatility.png",
        dpi=300,
    )

    plt.show()

    # ---------------------------------------------------------
    # Maximum Drawdown
    # ---------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.errorbar(
        timesteps,
        summary["mean_drawdown"],
        yerr=summary["std_drawdown"],
        marker="o",
        capsize=4,
    )

    plt.xscale("log")

    plt.xlabel("Training Timesteps")
    plt.ylabel("Maximum Drawdown")
    plt.title("Maximum Drawdown vs Training Timesteps")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "timestep_drawdown.png",
        dpi=300,
    )

    plt.show()


if __name__ == "__main__":
    run_timestep_experiment(
        rl_algorithm="PPO"
    )