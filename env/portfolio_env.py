import numpy as np

class PortfolioEnv:
    def __init__(self, windows, asset_names, initial_balance=1.0, risk_lambda=0.1):
        """
        windows: (T, window_size, num_features)
        asset_names: list of 5 tickers
        """
        self.windows = windows
        self.asset_names = asset_names
        self.T = len(windows)
        self.n_assets = len(asset_names)
        self.initial_balance = initial_balance
        self.risk_lambda = risk_lambda
        self.reset()

    def reset(self):
        self.t = 0
        self.wealth = self.initial_balance
        self.portfolio_returns = []
        return self._get_obs()

    def _get_obs(self):
        return self.windows[self.t]

    def _softmax(self, x):
        x = np.array(x)
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def step(self, action):
        # we must enforce weights>=0 and sum(weights)=1
        weights = self._softmax(action)
        current_window = self.windows[self.t] # today's window
        next_window = self.windows[self.t + 1]

        asset_returns = next_window[-1, :self.n_assets]
        portfolio_return = np.dot(weights, asset_returns) # e.g. 0.5TLT + 0.5SPY
        self.wealth *= np.exp(portfolio_return) # update wealth
        self.portfolio_returns.append(portfolio_return) # store for risk
        # compute reward
        reward = portfolio_return
        # risk penalty
        if len(self.portfolio_returns) > 20:
            vol = np.std(self.portfolio_returns[-20:])
            reward -= self.risk_lambda * vol
        self.t += 1
        done = self.t >= self.T - 2
        obs = self._get_obs() if not done else None
        info = {
            "wealth": self.wealth,
            "weights": weights,
            "portfolio_return": portfolio_return
        }
        return obs, reward, done, info
