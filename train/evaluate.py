"""
Evaluation utilities for RL portfolio models.

This module contains functions for evaluating the RL agents.
These include calculating portfolio performance metrics, comparing against baselines,
and analyzing portfolio behavior such as allocation, turnover, and
market regime sensitivity.
"""
import numpy as np

def evaluate_model(model, env):
    """
    Run a deterministic eval and collect portfolio history
    """
    obs, _ = env.reset()
    portfolio_values = []
    weights_history = []
    rewards = []
    daily_returns = []
    done = False
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    while not done:
        action, lstm_states = model.predict(
            obs, state=lstm_states,
            episode_start=episode_starts, deterministic=True,)
        obs, reward, terminated, truncated, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
        weights_history.append(info["weights"])
        rewards.append(reward)
        daily_returns.append(info["step_return"])
        done = terminated or truncated
        episode_starts = np.array([done], dtype=bool)
    return {
        "portfolio_values": np.array(portfolio_values),
        "weights": np.array(weights_history),
        "rewards": np.array(rewards),
        "daily_returns": np.array(daily_returns),
    }

def evaluate_baseline(returns, weights):
    """
    Evaluate a static portfolio allocation baseline.
    :return: portfolio performance metrics
    """
    step_returns = returns @ weights
    portfolio_values = np.exp(np.cumsum(step_returns))
    metrics = compute_metrics(
        portfolio_values,
        step_returns,
    )
    return metrics

def inspect_weights(model, env, n_steps=10):
    """
    Print portfolio allocation selected by model
    """
    obs, _ = env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    for _ in range(n_steps):
        action, lstm_states = model.predict(
            obs, state=lstm_states,
            episode_start=episode_starts, deterministic=True,)
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            "weights:",
            np.round(info["weights"], 3),
            "reward:",
            round(reward, 4)
        )
        done = terminated or truncated
        episode_starts = np.array([done], dtype=bool)
        if done:
            break

def compute_metrics(portfolio_values, daily_returns, risk_free_rate=0.0):
    """
    Calculate standard portfolio performance metrics.
    :param portfolio_values: Portfolio value history
    :param daily_returns: Daily portfolio returns
    :param risk_free_rate: Annual risk-free rate used for Sharpe ratio calculation.
    :return: Annual return, annual volatility, sharpe ratio, and maximum drawdown
    """
    annual_return = portfolio_values[-1] ** (252 / len(daily_returns)) - 1
    annual_vol = np.std(daily_returns) * np.sqrt(252)
    sharpe = (
        (annual_return - risk_free_rate) / annual_vol
        if annual_vol > 0 else 0.0
    )
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = portfolio_values / running_max - 1
    max_drawdown = np.min(drawdown)
    return {
        "Annual Return": annual_return,
        "Annual Volatility": annual_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown,
    }

def evaluate_and_compute_metrics(model, env):
    """
    Evaluate a model and return its performance metrics.
    """
    results = evaluate_model(model, env)
    metrics = compute_metrics(
        results["portfolio_values"],
        results["daily_returns"],
    )
    return metrics

def weight_statistics(weights):
    """
    Print summary statistics for portfolio weights
    """
    asset_names = ["SPY", "QQQ", "TLT", "GLD", "VNQ", "CASH"]
    print("\nWeight statistics")
    print("-" * 60)
    for i, asset in enumerate(asset_names):
        w = weights[:, i]
        print(
            f"{asset:4}"
            f" mean={w.mean():.3f}"
            f" std={w.std():.3f}"
            f" min={w.min():.3f}"
            f" max={w.max():.3f}"
        )

def turnover_statistics(weights):
    """
    Compute portfolio turnover.
    """
    turnover = np.abs(np.diff(weights, axis=0)).sum(axis=1)
    print("\nTurnover statistics")
    print("-" * 60)
    print(f"Mean   : {turnover.mean():.4f}")
    print(f"Median : {np.median(turnover):.4f}")
    print(f"Max    : {turnover.max():.4f}")
    return turnover

def equal_weight_distance(weights):
    """
    Compute equal weight distance
    """
    equal = np.ones(weights.shape[1]) / weights.shape[1]
    distance = np.abs(weights - equal).sum(axis=1)
    print("\nDistance to Equal Weight")
    print("-" * 60)
    print(f"Average L1 distance : {distance.mean():.4f}")
    print(f"Maximum distance    : {distance.max():.4f}")

def cash_statistics(weights, returns):
    """
    Analyze whether cash allocation predicts future returns.
    Last column = CASH.
    """
    cash = weights[:, -1]
    future_returns = returns[1:, :-1].mean(axis=1)
    # align lengths
    n = min(len(cash)-1, len(future_returns))
    cash = cash[:n]
    future_returns = future_returns[:n]
    high_cash = future_returns[cash > np.median(cash)]
    low_cash = future_returns[cash <= np.median(cash)]
    print("\nCash Timing Analysis")
    print("-" * 60)
    print(f"High cash periods future return: {high_cash.mean():.5f}")
    print(f"Low cash periods future return : {low_cash.mean():.5f}")

def regime_analysis(weights, returns):
    """
    Compare portfolio allocation during SPY up/down days.
    """
    min_len = min(len(weights), len(returns)) # Align the assets
    weights = weights[:min_len]
    returns = returns[:min_len]
    spy_returns = returns[:, 0]  # SPY is first asset
    bull_periods = spy_returns > 0
    bear_periods = spy_returns < 0
    bull_weights = weights[bull_periods]
    bear_weights = weights[bear_periods]
    print("\nBull market allocation")
    print("--------------------------------")
    print(np.mean(bull_weights, axis=0))
    print("\nBear market allocation")
    print("--------------------------------")
    print(np.mean(bear_weights, axis=0))
