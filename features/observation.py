"""
Shared construction of PPO observations.

Training and inference must build the same flattened vector: a market
feature window followed by the current portfolio weights.
"""
import numpy as np


def build_observation(market_window, current_weights):
    """
    Concatenate a market feature window with current portfolio weights.

    market_window : array-like
        Feature window with shape (window_size, n_features).
    current_weights : array-like
        Current portfolio allocation, length n_assets.
    Returns: 1-D float32 observation: flattened window then weights.
    """
    market_obs = np.asarray(market_window, dtype=np.float32)
    portfolio_state = np.asarray(current_weights, dtype=np.float32).reshape(-1)
    return np.concatenate([market_obs.flatten(), portfolio_state])
