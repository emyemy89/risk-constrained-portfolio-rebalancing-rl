"""
Visualization utilities for the experiments.

This module provides plotting functions for different goals, including comparing
portfolio performance, visualizing asset allocations, inspecting observation windows,
and analyzing cash allocation behavior over time.
"""
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

def plot_weights(weights):
    """
    Plot portfolio allocation changes over time.

    Creates a stacked area chart showing how the portfolio weights of each
    asset evolve during evaluation.
    """
    plt.figure(figsize=(10,5))
    plt.stackplot(range(len(weights)),weights.T, labels=["SPY", "QQQ", "TLT", "GLD", "VNQ", "CASH"])
    plt.title("PPO Portfolio Weights")
    plt.xlabel("Trading Days")
    plt.ylabel("Weight")
    plt.legend(["SPY", "QQQ", "TLT", "GLD", "VNQ", "CASH"])
    plt.grid(True)
    plt.show()


def inspect_observation(window):
    """
    Sanity check to ensure observation looks fine. Should be:
        Returns: noisy, oscillating around 0
        volatility: smoother, slowly changing.
        Momentum: smoother than returns, showing trends.
    :param window:
    :return: None
    """
    print("Shape:", window.shape)
    plt.figure(figsize=(12, 6))
    plt.plot(window)
    plt.title("Observation Window")
    plt.xlabel("Past Trading Days")
    plt.ylabel("Standardized Feature Value")
    plt.legend([
        "SPY_ret", "QQQ_ret", "TLT_ret", "GLD_ret", "VNQ_ret",
        "SPY_vol", "QQQ_vol", "TLT_vol", "GLD_vol", "VNQ_vol",
        "SPY_mom", "QQQ_mom", "TLT_mom", "GLD_mom", "VNQ_mom",
    ])
    plt.grid(True)
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
