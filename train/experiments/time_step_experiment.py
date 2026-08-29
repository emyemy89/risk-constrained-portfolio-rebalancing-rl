"""
Experiment for different values of timesteps used in PPO agent training
"""
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
import torch

from data.pipeline import load_test_data
from train.make_env import make_env
from train.utils.algorithm_selection import create_model
from train.utils.run_info import run_debugging_info
from data.extract.load_data import ASSET_NAMES


TIMESTEPS = [
    10_000,
    50_000,
    100_000,
    150_000,
    200_000,
    300_000,
    500_000,
    800_000,
    2_000_000,
    10_000_000,
]

SEED = 0


# -------------------------------------------------------------------
# Load data once
# -------------------------------------------------------------------

train_windows, train_returns, test_windows, test_returns = load_test_data()


results = []


# -------------------------------------------------------------------
# Train and evaluate one independent model per timestep
# -------------------------------------------------------------------

for total_timesteps in TIMESTEPS:

    print(f"\n===== {total_timesteps:,} timesteps =====")

    # Reset all random generators so every experiment starts
    # from the same random state.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Create completely fresh environments
    train_env = make_env(train_windows, train_returns)
    test_env = make_env(test_windows, test_returns)

    train_env.reset(seed=SEED)
    test_env.reset(seed=SEED)

    # Create a completely fresh model
    model = create_model(
        rl_algorithm="PPO",
        train_env=train_env,
        seed=SEED,
    )

    # Train only for the requested number of timesteps
    model.learn(total_timesteps=total_timesteps)

    # Evaluate on the same untouched test set
    metrics = run_debugging_info(
        model,
        test_env,
        test_returns,
        seed=SEED,
        fold_idx=f"timesteps_{total_timesteps}",
        asset_names=ASSET_NAMES,
    )

    results.append({
        "timesteps": total_timesteps,
        "Annual Return": metrics["Annual Return"],
        "Annual Volatility": metrics["Annual Volatility"],
        "Sharpe Ratio": metrics["Sharpe Ratio"],
        "Max Drawdown": metrics["Max Drawdown"],
    })

    # Explicitly close environments
    train_env.close()
    test_env.close()


# -------------------------------------------------------------------
# Results
# -------------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n===== Results =====")
print(results_df.to_string(index=False))


# -------------------------------------------------------------------
# Plots
# -------------------------------------------------------------------

metrics_to_plot = [
    ("Annual Return", "Annual Return vs Training Timesteps"),
    ("Annual Volatility", "Annual Volatility vs Training Timesteps"),
    ("Sharpe Ratio", "Sharpe Ratio vs Training Timesteps"),
    ("Max Drawdown", "Maximum Drawdown vs Training Timesteps"),
]

for metric, title in metrics_to_plot:
    plt.figure()
    plt.plot(
        results_df["timesteps"],
        results_df[metric],
        marker="o",
    )
    plt.xscale("log")
    plt.xlabel("Total Timesteps")
    plt.ylabel(metric)
    plt.title(title)
    plt.grid()
    plt.show()
