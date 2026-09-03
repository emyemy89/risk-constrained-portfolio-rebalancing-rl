"""
Unit tests for portfolio_env.
"""
import numpy as np

from env.portfolio_env import PortfolioEnv


def make_test_env(
    n_timesteps=10,
    window_size=3,
    n_features=2,
    n_assets=3,
    transaction_cost=0.001,
    reward_horizon=1,
):
    """
    Create a small deterministic environment for unit testing.
    """
    windows = np.zeros((n_timesteps, window_size, n_features), dtype=np.float32,)
    returns = np.zeros(
        (n_timesteps, n_assets),
        dtype=np.float64,
    )
    return PortfolioEnv(
        windows=windows,
        returns=returns,
        initial_cash=1.0,
        risk_lambda=0.0,
        volatility_window=3,
        transaction_cost=transaction_cost,
        reward_horizon=reward_horizon,
    )


def test_reset_returns_valid_observation():
    """
    Test that after resset we get a valid observation
    """
    env = make_test_env()
    observation, info = env.reset()
    assert observation.shape == env.observation_space.shape
    assert observation.dtype == np.float32
    assert np.all(np.isfinite(observation))
    assert env.observation_space.contains(observation)


def test_reset_initializes_valid_weights():
    """
    Test that after resset we get a valid weights
    """
    env = make_test_env(n_assets=3)
    env.reset()
    weights = env.prev_weights
    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 1.0)
    assert np.all(np.isfinite(weights))


def test_weights_sum_to_one():
    """
    Test that sum adds up to 1
    """
    env = make_test_env(n_assets=3)
    env.reset()
    action = np.array([1.0, -1.0, 0.5], dtype=np.float32)
    _, _, _, _, info = env.step(action)
    weights = info["weights"]
    assert np.isclose(np.sum(weights), 1.0)


def test_weights_are_bounded():
    """
    Test that weights are bounded between 0 and 1
    """
    env = make_test_env(n_assets=3)
    env.reset()
    actions = [
        np.array([1.0, 1.0, 1.0], dtype=np.float32),
        np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        np.array([1.0, -1.0, 1.0], dtype=np.float32),
    ]
    for action in actions:
        _, _, _, _, info = env.step(action)
        weights = info["weights"]
        assert np.all(weights >= 0.0)
        assert np.all(weights <= 1.0)


def test_max_weight_change_is_respected():
    """
    Test that max_weight_change changes
    """
    env = make_test_env(n_assets=3)
    env.reset()
    previous_weights = env.prev_weights.copy()
    action = np.array([1.0, -1.0, 1.0], dtype=np.float32)
    _, _, _, _, info = env.step(action)
    weights = info["weights"]
    weight_changes = np.abs(weights - previous_weights)
    assert np.all(weight_changes <= env.max_weight_change + 1e-8)

def test_no_nan_weights():
    """
    Test that no nan weights
    """
    env = make_test_env(n_assets=3)
    env.reset()
    actions = [
        np.array([1.0, 1.0, 1.0], dtype=np.float32),
        np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        np.array([1.0, -1.0, 0.0], dtype=np.float32),
        np.array([0.5, -0.7, 1.0], dtype=np.float32),
    ]
    for action in actions:
        _, _, _, _, info = env.step(action)
        weights = info["weights"]
        assert np.all(np.isfinite(weights))

def test_transaction_cost_applied():
    """
    Test that transaction_cost applies
    """
    transaction_cost = 0.001
    env = make_test_env(
        n_assets=3,
        transaction_cost=transaction_cost,
    )
    env.reset()
    previous_weights = env.prev_weights.copy()
    action = np.array([1.0, -1.0, 0.0], dtype=np.float32)
    _, _, _, _, info = env.step(action)
    weights = info["weights"]
    turnover = 0.5 * np.sum(
        np.abs(weights - previous_weights)
    )
    expected_cost = transaction_cost * turnover
    expected_portfolio_value = 1.0 * (1.0 - expected_cost)
    assert np.isclose(
        env.portfolio_value,
        expected_portfolio_value,
    )

def test_step_returns_valid_observation():
    """
    Check that the observation remains valid after a transition, not only after reset
    """
    env = make_test_env()
    env.reset()
    action = np.array([0.5, -0.5, 0.0], dtype=np.float32)
    observation, _, _, _, _ = env.step(action)
    assert observation.shape == env.observation_space.shape
    assert observation.dtype == np.float32
    assert np.all(np.isfinite(observation))
    assert env.observation_space.contains(observation)

def test_portfolio_value_updates_with_returns():
    """
    Test portfolio evaluation
    """
    returns = np.array([
        [0.0, 0.0, 0.0],
        [0.01, 0.02, 0.03],
        [0.0, 0.0, 0.0],
    ])
    windows = np.zeros(
        (3, 3, 2),
        dtype=np.float32,
    )
    env = PortfolioEnv(
        windows=windows,
        returns=returns,
        initial_cash=1.0,
        risk_lambda=0.0,
        volatility_window=3,
        transaction_cost=0.0,
        reward_horizon=1,
    )
    env.reset()
    action = np.zeros(3, dtype=np.float32)
    _, _, _, _, info = env.step(action)
    expected_simple_returns = np.expm1([0.01, 0.02, 0.03])
    expected_return = np.mean(expected_simple_returns)
    expected_value = 1.0 + expected_return
    assert np.isclose(
        env.portfolio_value,
        expected_value,
    )

def test_episode_terminates_at_expected_step():
    """
    Test that the episode terminates at the expected step
    """
    env = make_test_env(
        n_timesteps=10,
        reward_horizon=1,
    )
    env.reset()
    terminated = False
    for _ in range(20):
        action = np.zeros(3, dtype=np.float32)
        _, _, terminated, _, _ = env.step(action)
        if terminated:
            break
    assert terminated
