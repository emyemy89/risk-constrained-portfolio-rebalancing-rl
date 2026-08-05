"""
Portfolio's environment.

This module defines a Gymnasium-compatible environment for training RL agents to dynamically
rebalance the multi-asset portfolio.
The environment exposes historical market observations and portfolio allocations as states. It
accepts allocation adjustment actions. The rewards are returned based on future portfolio
performance, transaction costs, and optional risk penalties.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    """
    The agent observes a rolling window of market features together with the
    current portfolio weights and outputs allocation adjustments for each asset.
    The environment updates the portfolio, calculates returns, applies
    transaction costs and risk penalties, and provides a reward signal based
    on future portfolio performance.

    The action space represents relative changes to current portfolio weights.
    At the same time the resulting weights are clipped to valid allocations and
    normalized to maintain a fully invested portfolio.

    Observation:
        Flattened historical market window concatenated with current asset
        allocation weights.

    Action:
        Continuous allocation adjustments for each asset in the range [-1, 1].
        The actual weight change is limited by `max_weight_change`.

    Reward:
        Future portfolio return over `reward_horizon` steps, adjusted for:
        - transaction costs based on portfolio turnover
        - additional penalty for negative daily returns
        - optional volatility-based risk penalty

    Args:
        windows(np.array): Rolling market feature windows with shape
                           (timesteps, window_size, n_features).

        returns(np.array): Asset return matrix with shape (timesteps, n_assets).

        initial_cash(float): Initial portfolio value.

        risk_lambda(float): Weight of the volatility penalty in the reward.

        volatility_window(int): Number of past returns used to estimate portfolio volatility.

        transaction_cost(float):Cost applied to portfolio turnover.

        reward_horizon (int): Number of future steps used when calculating rewards.
        """
    def __init__(
            self,
            windows,
            returns,
            initial_cash=1.0,
            risk_lambda=0.00,
            volatility_window=20,
            transaction_cost=0.0005,
            reward_horizon=10,
    ):
        self.windows = windows
        self.returns = returns
        self.n_assets = self.returns.shape[1]
        self.reward_horizon = reward_horizon

        self.initial_cash = initial_cash
        self.risk_lambda = risk_lambda
        self.volatility_window = volatility_window
        self.transaction_cost = transaction_cost

        self.current_step = 0
        self.portfolio_value = initial_cash
        self.prev_weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_returns = []
        self.initial_value = initial_cash

        # Action
        self.max_weight_change = 0.2 # Do not go more than 20% in allocation in one step
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.n_assets,),
            dtype=np.float32,
        )

        # Observation
        obs_size = (
                self.windows.shape[1] *
                self.windows.shape[2]
                + self.n_assets
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,), # (3000, 19, 30) -> (19 x 30)+5
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_value = self.initial_cash
        self.prev_weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_returns = []
        self.initial_value = self.portfolio_value
        return self._get_obs(), {}

    def _get_obs(self):
        market_obs = self.windows[self.current_step].astype(np.float32)
        portfolio_state = self.prev_weights.astype(np.float32)
        return np.concatenate([market_obs.flatten(),portfolio_state])

    def step(self, action):
        # (St, action) -> (St+1, reward)
        delta_weights = action * self.max_weight_change # action represents allocation changes
        weights = self.prev_weights + delta_weights
        weights = np.clip(weights, 0, 1)  # enforce valid portfolio weights
        weights /= np.sum(weights) # normalize
        # Use Turnover=1/2 ∑ ∣ w_{i,t} −w{i,t-1} ∣
        turnover = 0.5*np.sum(np.abs(weights - self.prev_weights))
        # One-day return (used for portfolio evolution)
        next_returns = self.returns[self.current_step + 1]
        portfolio_return = np.dot(weights, next_returns)
        self.portfolio_value *= np.exp(portfolio_return)
        self.portfolio_returns.append(portfolio_return)
        # Calculate return for a set horizon
        future_returns = self.returns[
            self.current_step + 1:
            self.current_step + 1 + self.reward_horizon
        ]

        # Compute reward
        reward = np.sum(future_returns @ weights)
        # Make losses more costly
        if portfolio_return < 0:
            reward += 0.5 * portfolio_return
        reward -= self.transaction_cost * turnover

        # risk penalty
        if len(self.portfolio_returns) >= self.volatility_window:
            recent_returns = self.portfolio_returns[-self.volatility_window:]
            reward -= self.risk_lambda * np.std(recent_returns)
        # Move to next step
        self.current_step += 1
        terminated = self.current_step >=len(self.windows) - self.reward_horizon - 1
        next_obs = self._get_obs()
        self.prev_weights = weights
        episode_return = self.portfolio_value / self.initial_value - 1
        info = {
            "portfolio_value": self.portfolio_value,
            "weights": weights,
            "cumm_return": self.portfolio_value / self.initial_value - 1,
            "episode_return": episode_return,
            "step_return": portfolio_return,
        }
        return (
            next_obs,
            reward,
            terminated,
            False,  # truncated
            info
        )
