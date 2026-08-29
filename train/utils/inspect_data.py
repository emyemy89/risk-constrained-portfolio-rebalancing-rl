"""
Visualization utilities for the experiments.

This module provides plotting functions for different goals, including comparing
portfolio performance, visualizing asset allocations, inspecting observation windows,
and analyzing cash allocation behavior over time.
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_portfolios(strategy_values):
    """
    Plot portfolio value evolution for multiple strategies.
    """
    plt.figure(figsize=(10, 5))
    for name, values in strategy_values.items():
        plt.plot(values, label=name)
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value")
    plt.title("Portfolio Performance Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_weights(weights, asset_names):
    """
    Plot portfolio allocation changes over time.

    Creates a stacked area chart showing how the portfolio weights of each
    asset evolve during evaluation.
    """
    plt.figure(figsize=(10,5))
    plt.stackplot(range(len(weights)),weights.T, labels=asset_names)
    plt.title("PPO Portfolio Weights")
    plt.xlabel("Trading Days")
    plt.ylabel("Weight")
    plt.legend(asset_names)
    plt.grid(True)
    plt.show()


def inspect_observation(window, feature_columns):
    """
    Visualize an observation window in a compact grid, grouped by
    feature family.
    Parameters
    ----------
    window : np.ndarray
        Observation window of shape (window_size, n_features).

    feature_columns : pd.Index or pd.MultiIndex
        Columns corresponding to the features in `window`.
    """
    print("Observation shape:", window.shape)
    # Create readable feature labels
    labels = [
        " | ".join(map(str, col)) if isinstance(col, tuple) else str(col)
        for col in feature_columns
    ]
    # Group feature indices by top-level feature family
    groups = {}
    for i, col in enumerate(feature_columns):
        if isinstance(col, tuple):
            group = str(col[0])
        else:
            group = "features"
        groups.setdefault(group, []).append(i)
    # Create compact subplot grid
    n_groups = len(groups)
    n_cols = 3
    n_rows = int(np.ceil(n_groups / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(18, 4 * n_rows), squeeze=False
    )
    axes = axes.flatten()

    # Plot each feature family
    for ax, (group, indices) in zip(axes, groups.items()):
        for i in indices:
            ax.plot(window[:, i], label=labels[i], linewidth=1)
        ax.set_title(group)
        ax.set_xlabel("Past Trading Days")
        ax.set_ylabel("Standardized Value")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    # Hide unused subplots
    for ax in axes[n_groups:]:
        ax.set_visible(False)
    fig.suptitle("Observation Window — Feature Overview", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_cash_weight(weights):
    """
    Plot cash allocation over time
    weights shape: (timesteps, assets)
    Last column is CASH.
    """
    cash_weights = weights[:, -1]
    plt.figure(figsize=(10, 4))
    plt.plot(cash_weights)
    plt.title("Cash Allocation Over Time")
    plt.xlabel("Trading Days")
    plt.ylabel("Cash Weight")
    plt.grid(True)
    plt.show()
