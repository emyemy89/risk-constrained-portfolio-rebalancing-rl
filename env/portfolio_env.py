import numpy as np

class PortfolioEnv:
    def __init__(self, windows, returns, initial_capital=1.0):
        """
        windows: (T, window_size, num_features)
        returns: (T, num_assets)
        """
        self.windows = windows
        self.returns = returns
        self.initial_capital = initial_capital

        self.T = windows.shape[0]
        self.num_assets = returns.shape[1]

        self.reset()

    def reset(self):
        self.t = 0
        self.wealth = self.initial_capital
        self.done = False

        return self._get_obs()

    def _get_obs(self):
        return self.windows[self.t]