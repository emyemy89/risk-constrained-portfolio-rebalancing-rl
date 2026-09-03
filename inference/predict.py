"""
Minimal inference for the trained PPO portfolio-allocation agent.

This module orchestrates existing data, feature, observation, and
action-to-weight logic. It does not retrain the model.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from data.extract.load_data import ETF_TICKERS
from data.pipeline import (DEFAULT_OBS_WINDOW_SIZE, load_latest_observation,)
from env.portfolio_env import DEFAULT_MAX_WEIGHT_CHANGE, action_to_weights
from features.observation import build_observation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "final_model.zip"


def _resolve_weights(current_weights, asset_names):
    """Convert caller weights into a 1-D array in training asset order."""
    n_assets = len(asset_names)
    if isinstance(current_weights, dict):
        missing = [name for name in asset_names if name not in current_weights]
        extra = [name for name in current_weights if name not in asset_names]
        if missing or extra:
            raise ValueError(
                "current_weights keys must match assets "
                f"{asset_names}; missing={missing}, extra={extra}"
            )
        weights = np.array(
            [current_weights[name] for name in asset_names],dtype=np.float64)
    else:
        weights = np.asarray(current_weights, dtype=np.float64).reshape(-1)
        if weights.shape[0] != n_assets:
            raise ValueError(
                f"current_weights must have length {n_assets} "
                f"(assets={asset_names}), got {weights.shape[0]}"
            )
    if not np.all(np.isfinite(weights)):
        raise ValueError("current_weights must be finite")
    return weights


def predict_allocation(
    current_weights, *, model_path=None,
    market_data=None, model=None):
    """
    Recommend a portfolio allocation from the frozen PPO model.

    Parameters:
    current_weights : array-like or dict
        Current allocation. A sequence is interpreted in the project's asset
        order. A dict must be keyed by those asset names.
    model_path : path-like, optional
        Location of ``final_model.zip``. Defaults to ``models/final_model.zip``.
    market_data : pd.DataFrame, optional
        Pre-loaded prices (for tests). When omitted, latest Yahoo Finance
        data is downloaded.
    model : optional
        Pre-loaded Stable-Baselines3 model. When omitted, PPO is loaded from
        ``model_path``. Provided so tests can avoid the trained zip file.

    Returns:
    dict
        ``date``, ``weights`` (asset name to float), and raw PPO ``action``.
    """
    window, as_of_date, asset_names = load_latest_observation(
        obs_window_size=DEFAULT_OBS_WINDOW_SIZE, market_data=market_data)
    weights = _resolve_weights(current_weights, asset_names)
    observation = build_observation(window, weights)

    if model is None:
        path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        model = PPO.load(str(path))

    action, _ = model.predict(observation, deterministic=True)
    recommended = action_to_weights(action, weights,
        max_weight_change=DEFAULT_MAX_WEIGHT_CHANGE)
    as_of = pd.Timestamp(as_of_date)
    return {
        "date": as_of.date().isoformat(),
        "weights": {
            name: float(value)
            for name, value in zip(asset_names, recommended)
        },
        "action": np.asarray(action, dtype=np.float32),
    }


def _example_equal_weights(n_assets):
    """Return a fully invested equal-weight allocation."""
    return np.ones(n_assets, dtype=np.float64) / n_assets


def main():
    """Download latest data and print a sample allocation recommendation."""
    n_assets = len(ETF_TICKERS) + 1  # synthetic CASH
    current_weights = _example_equal_weights(n_assets)
    result = predict_allocation(current_weights)
    print(f"As of {result['date']}")
    print("Recommended weights:")
    for name, weight in result["weights"].items():
        print(f"  {name:5} {weight:.4f}")
    print("Raw action:", np.round(result["action"], 4))


if __name__ == "__main__":
    main()
