import numpy as np
import gymnasium as gym

from gymnasium import spaces


class PortfolioEnv(gym.Env):
    def __init__(
            self,
            windows,
            returns,
            initial_cash=1.0,
            risk_lambda=0.1,
            volatility_window=20,
    ):
        self.windows = windows
        self.returns = returns
        self.n_assets = self.returns.shape[1]

        self.initial_cash = initial_cash
        self.risk_lambda = risk_lambda
        self.volatility_window = volatility_window

        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_assets,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.windows.shape[1:],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_value = self.initial_cash
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_returns = []
        observation = self.windows[self.current_step]
        info = {}
        return observation, info

    def _get_obs(self):
        return self.windows[self.current_step]

    def _softmax(self, x):
        x = np.array(x)
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def step(self, action):
        # we must enforce weights>=0 and sum(weights)=1
        weights = self._softmax(action)
        next_returns = self.returns[self.current_step + 1]
        portfolio_return = np.dot(weights, next_returns) # e.g. 0.5TLT + 0.5SPY
        self.portfolio_value *= np.exp(portfolio_return) # update wealth
        self.portfolio_returns.append(portfolio_return) # store for risk
        # compute reward
        reward = portfolio_return
        # risk penalty
        if len(self.portfolio_returns) >= self.volatility_window:
            recent_returns = self.portfolio_returns[-self.volatility_window:]
            vol = np.std(recent_returns)
            reward -= self.risk_lambda * vol
        self.current_step += 1
        terminated = self.current_step >= len(self.windows) - 2
        obs = self._get_obs() if not terminated else None
        info = {
            "portfolio_value": self.portfolio_value,
            "weights": weights,
            "portfolio_return": portfolio_return
        }
        return (
            obs,
            reward,
            terminated,
            False,  # truncated
            info
        )
