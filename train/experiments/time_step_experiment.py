"""
Experiment: effect of PPO training timesteps.

Each timestep is trained with multiple random seeds and evaluated
on the same validation period.
"""

import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from data.pipeline import load_data
from train.make_env import make_env
from train.utils.algorithm_selection import create_model
from train.evaluate import evaluate_and_compute_metrics


TIMESTEPS = [
    100_000,
    500_000,
    2_000_000,
]

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


# ---------------------------------------------------------
# Load one fixed validation split
# ---------------------------------------------------------

(
    train_windows,
    train_returns,
    val_windows,
    val_returns,
    feature_columns,
) = load_data(
    train_end="2018-12-31",
    val_start="2019-01-01",
    val_end="2021-12-31",
)


results = []


# ---------------------------------------------------------
# Run every timestep with every seed
# ---------------------------------------------------------

for total_timesteps in TIMESTEPS:

    for seed in SEEDS:

        print(
            f"\n===== {total_timesteps:,} timesteps | "
            f"seed {seed} ====="
        )

        # Reset random state
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Fresh environments
        train_env = make_env(
            train_windows,
            train_returns,
        )

        val_env = make_env(
            val_windows,
            val_returns,
        )

        train_env.reset(seed=seed)
        val_env.reset(seed=seed)

        # Fresh PPO model
        model = create_model(
            rl_algorithm="PPO",
            train_env=train_env,
            seed=seed,
        )

        # Train
        model.learn(
            total_timesteps=total_timesteps
        )

        # Evaluate on validation set
        metrics = evaluate_and_compute_metrics(
            model,
            val_env,
        )

        results.append({
            "timesteps": total_timesteps,
            "seed": seed,
            "Annual Return": metrics["Annual Return"],
            "Annual Volatility": metrics["Annual Volatility"],
            "Sharpe Ratio": metrics["Sharpe Ratio"],
            "Max Drawdown": metrics["Max Drawdown"],
        })

        train_env.close()
        val_env.close()


# ---------------------------------------------------------
# Raw results
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n===== Individual Results =====")
print(results_df.to_string(index=False))


# ---------------------------------------------------------
# Aggregate across seeds
# ---------------------------------------------------------

summary = (
    results_df
    .groupby("timesteps")
    .agg({
        "Annual Return": ["mean", "std"],
        "Annual Volatility": ["mean", "std"],
        "Sharpe Ratio": ["mean", "std"],
        "Max Drawdown": ["mean", "std"],
    })
)

print("\n===== Mean ± Std Across Seeds =====")
print(summary)


# ---------------------------------------------------------
# Plot mean performance
# ---------------------------------------------------------

mean_results = (
    results_df
    .groupby("timesteps")
    .mean(numeric_only=True)
    .reset_index()
)


metrics_to_plot = [
    ("Annual Return", "Mean Annual Return"),
    ("Annual Volatility", "Mean Annual Volatility"),
    ("Sharpe Ratio", "Mean Sharpe Ratio"),
    ("Max Drawdown", "Mean Maximum Drawdown"),
]


for metric, title in metrics_to_plot:

    plt.figure()

    plt.plot(
        mean_results["timesteps"],
        mean_results[metric],
        marker="o",
    )

    plt.xscale("log")
    plt.xlabel("Training Timesteps")
    plt.ylabel(metric)
    plt.title(title)
    plt.grid()

    plt.show()
