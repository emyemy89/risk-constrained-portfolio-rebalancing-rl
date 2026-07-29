import numpy as np
import gymnasium as gym
import joblib

from gymnasium import spaces


class PortfolioEnv(gym.Env):
    def __init__(
            self,
            windows,
            returns,
            ranking_probs,
            initial_cash=1.0,
            risk_lambda=0.00,
            volatility_window=20,
            transaction_cost=0.001,
    ):
        self.windows = windows
        self.returns = returns
        self.ranking_probs = ranking_probs
        self.n_assets = self.returns.shape[1]

        self.initial_cash = initial_cash
        self.risk_lambda = risk_lambda
        self.volatility_window = volatility_window
        self.transaction_cost = transaction_cost

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
        ranking_probabilities = self.ranking_probs[self.current_step]
        return np.concatenate([
            market_obs.flatten(),
            portfolio_state,
            ranking_probabilities
        ])

    def step(self, action):
        # (St, action) -> (St+1, reward)
        delta_weights = action * self.max_weight_change # action represents allocation changes
        weights = self.prev_weights + delta_weights
        weights = np.clip(weights, 0, 1)  # enforce valid portfolio weights
        weights /= np.sum(weights) # normalize
        turnover = np.sum(np.abs(weights - self.prev_weights))
        next_returns = self.returns[self.current_step + 1]
        portfolio_return = np.dot(weights, next_returns) # e.g. 0.5TLT + 0.5SPY
        self.portfolio_value *= np.exp(portfolio_return) # update wealth
        self.portfolio_returns.append(portfolio_return) # store for risk

        # compute reward
        reward = portfolio_return
        reward -= self.transaction_cost * turnover
        

        # risk penalty
        if len(self.portfolio_returns) >= self.volatility_window:
            recent_returns = self.portfolio_returns[-self.volatility_window:]
            reward -= self.risk_lambda * np.std(recent_returns)
        self.current_step += 1
        terminated = self.current_step >= len(self.windows) - 2
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
