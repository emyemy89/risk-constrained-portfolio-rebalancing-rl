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