"""
Unit tests for inference observation construction and allocation output.
"""
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from stable_baselines3 import PPO

from data.extract.load_data import ETF_TICKERS
from data.pipeline import DEFAULT_OBS_WINDOW_SIZE, load_latest_observation
from env.portfolio_env import (
    DEFAULT_MAX_WEIGHT_CHANGE,
    PortfolioEnv,
    action_to_weights,
)
from features.observation import build_observation
from inference.predict import DEFAULT_MODEL_PATH, predict_allocation


def make_synthetic_etf_data(start="2018-01-01", end="2026-06-01", seed=0):
    """Yahoo Finance-style Close prices covering the training cutoff."""
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    frames = {}
    for ticker in ETF_TICKERS:
        log_returns = rng.normal(0.0002, 0.01, size=len(dates))
        frames[ticker] = 100.0 * np.exp(np.cumsum(log_returns))
    close = pd.DataFrame(frames, index=dates)
    close.columns = pd.MultiIndex.from_product([["Close"], list(ETF_TICKERS)])
    return close


class FrozenPolicy:  # pylint: disable=too-few-public-methods
    """Stand-in PPO policy that never trains and returns a fixed action."""

    def __init__(self, n_assets, observation_shape):
        self.n_assets = n_assets
        self.observation_shape = observation_shape
        self.learn = MagicMock(side_effect=AssertionError("model.learn must not run"))
        self.action = np.linspace(-0.5, 0.5, n_assets).astype(np.float32)

    def predict(self, observation, deterministic=True):
        """Return a deterministic dummy action after checking observation shape."""
        observation = np.asarray(observation)
        assert observation.shape == self.observation_shape
        assert deterministic is True
        return self.action, None


@pytest.fixture(name="market_data")
def fixture_market_data():
    """Synthetic Yahoo Finance Close prices spanning the training cutoff."""
    return make_synthetic_etf_data()


def test_inference_observation_matches_environment(market_data):
    """Inference observation must match the environment observation vector."""
    window, _, asset_names = load_latest_observation(market_data=market_data)
    n_assets = len(asset_names)
    current_weights = np.ones(n_assets) / n_assets
    observation = build_observation(window, current_weights)

    env = PortfolioEnv(
        windows=window[np.newaxis, ...],
        returns=np.zeros((2, n_assets)),
    )
    env_observation, _ = env.reset()

    assert observation.shape == env.observation_space.shape
    assert observation.dtype == np.float32
    np.testing.assert_array_equal(observation, env_observation)
    assert env.observation_space.contains(observation)


def test_output_contains_all_assets_and_valid_weights(market_data):
    """Recommended weights include every asset and satisfy env constraints."""
    window, as_of_date, asset_names = load_latest_observation(market_data=market_data)
    n_assets = len(asset_names)
    observation = build_observation(window, np.ones(n_assets) / n_assets)
    model = FrozenPolicy(n_assets, observation.shape)

    current = {name: 1.0 / n_assets for name in asset_names}
    result = predict_allocation(
        current,
        market_data=market_data,
        model=model,
    )

    assert set(result["weights"]) == set(asset_names)
    assert list(result["weights"]) == asset_names
    weights = np.array(list(result["weights"].values()), dtype=np.float64)
    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 1.0)
    assert np.all(np.isfinite(weights))
    prev = np.array([current[name] for name in asset_names])
    assert np.all(
        np.abs(weights - prev) <= DEFAULT_MAX_WEIGHT_CHANGE + 1e-8
    )
    expected = action_to_weights(model.action, prev)
    np.testing.assert_allclose(weights, expected)
    assert result["date"] == pd.Timestamp(as_of_date).date().isoformat()
    model.learn.assert_not_called()


def test_inference_does_not_call_learn(market_data, monkeypatch):
    """Loading and predicting must not invoke PPO.learn."""
    window, _, asset_names = load_latest_observation(market_data=market_data)
    n_assets = len(asset_names)
    observation = build_observation(window, np.ones(n_assets) / n_assets)
    frozen = FrozenPolicy(n_assets, observation.shape)

    def fake_load(_path):
        return frozen

    monkeypatch.setattr("inference.predict.PPO.load", fake_load)
    predict_allocation(
        np.ones(n_assets) / n_assets,
        market_data=market_data,
        model_path=DEFAULT_MODEL_PATH,
    )
    frozen.learn.assert_not_called()


def test_latest_window_does_not_use_future_dates(market_data):
    """The observation window must end on or before the last available price date."""
    window, as_of_date, _ = load_latest_observation(market_data=market_data)
    assert window.shape[0] == DEFAULT_OBS_WINDOW_SIZE
    assert as_of_date <= market_data.index.max()


@pytest.mark.skipif(
    not DEFAULT_MODEL_PATH.exists(),
    reason="trained model zip is not present",
)
def test_observation_matches_saved_ppo_space(market_data):
    """When the trained zip is present, the observation must match its space."""
    window, _, asset_names = load_latest_observation(market_data=market_data)
    observation = build_observation(
        window,
        np.ones(len(asset_names)) / len(asset_names),
    )
    trained = PPO.load(str(DEFAULT_MODEL_PATH))
    assert observation.shape == trained.observation_space.shape
    assert trained.observation_space.contains(observation)


def test_action_to_weights_respects_environment_constraints():
    """Action projection must stay on the simplex within the max step size."""
    prev = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    action = np.array([1.0, -1.0, 0.5, 0.0, -0.25])
    weights = action_to_weights(action, prev)
    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 1.0)
    assert np.all(np.abs(weights - prev) <= DEFAULT_MAX_WEIGHT_CHANGE + 1e-8)
    assert np.all(np.isfinite(weights))
