"""
Momentum-based portfolio baseline.

This module implements a simple trend-following strategy and performance
evaluation utilities. The momentum strategy selects the asset with the
strongest historical return over a fixed period and then allocates the
entire portfolio to that asset for the following period.
"""
import numpy as np


def run_momentum_strategy(returns, lookback=20):
    """
   Run a simple momentum-based asset allocation strategy.

   At each timestep, the strategy ranks assets based on their cumulative
   returns over the previous lookback period and invests entirely in the
   asset with the highest momentum.
   """
    portfolio_returns = []
    for t in range(lookback, len(returns)):
        # past momentum
        past_returns = returns[t-lookback:t]
        momentum = np.sum(past_returns, axis=0)
        # choose best asset
        best_asset = np.argmax(momentum)
        weights = np.zeros(returns.shape[1])
        weights[best_asset] = 1.0
        # next day return
        portfolio_return = np.dot(weights, returns[t])
        portfolio_returns.append(portfolio_return)
    return np.array(portfolio_returns)

def calculate_metrics(portfolio_returns):
    """
    Calculate portfolio performance metrics.

    Computes annualized return, annualized volatility, sharpe ratio, and
    maximum drawdown from a series of portfolio log returns.
    """
    annual_return = (
        np.exp(np.mean(portfolio_returns) * 252) - 1
    )
    annual_vol = np.std(portfolio_returns) * np.sqrt(252)
    sharpe = annual_return / annual_vol
    cumulative_curve = np.exp(np.cumsum(portfolio_returns))
    running_max = np.maximum.accumulate(cumulative_curve)
    drawdown = cumulative_curve / running_max - 1
    max_drawdown = np.min(drawdown)
    return {
        "Annual Return": annual_return,
        "Annual Volatility": annual_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown,
    }
