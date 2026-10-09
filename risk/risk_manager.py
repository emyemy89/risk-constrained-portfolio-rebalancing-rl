"""
Risk Manager
"""
import numpy as np

class RiskManager:
    """
    The risk manager is meant to stop 3 wrong executions in a row
    """
    def __init__(self, rolling_window=5,
                 max_consecutive_losses=3, recovery_days=3,
                 max_rolling_loss=0.05, max_drawdown=0.08,):
        self.max_consecutive_losses = max_consecutive_losses
        self.rolling_window = rolling_window
        self.max_rolling_loss = max_rolling_loss
        self.max_drawdown = max_drawdown
        self.recovery_days = recovery_days
        self.reset()

    def reset(self):
        self.consecutive_losses = 0
        self.recovery_count = 0
        self.return_history = []
        self.peak_value = None
        self.stopped = False
        self.stop_reason = None

    def record_return(self, daily_return, portfolio_value):
        """
        Record the daily return value
        """
        if self.stopped:
            return
        if daily_return < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        self.return_history.append(daily_return)
        self.return_history = self.return_history[-self.rolling_window:]

        if self.peak_value is None:
            self.peak_value = portfolio_value
        else:
            self.peak_value = max(self.peak_value, portfolio_value)
        drawdown = portfolio_value / self.peak_value - 1

        rolling_return = None
        if len(self.return_history) == self.rolling_window:
            rolling_return = (np.prod(1 + np.array(self.return_history)) - 1)

        if self.consecutive_losses >= self.max_consecutive_losses:
            self.stopped = True
            self.stop_reason = "consecutive_losses"

        elif rolling_return is not None and rolling_return <= -self.max_rolling_loss:
            self.stopped = True
            self.stop_reason = "rolling_loss"

        elif drawdown <= -self.max_drawdown:
            self.stopped = True
            self.stop_reason = "drawdown"

        if self.stopped:
            self.recovery_count = 0

    def update_recovery(self, market_recovered):
        """
        Update the risk manager with market recovered
        """
        if not self.stopped:
            return
        if market_recovered:
            self.recovery_count += 1
        else:
            self.recovery_count = 0
        if self.recovery_count >= self.recovery_days:
            self.stopped = False
            self.stop_reason = None
            self.consecutive_losses = 0
            self.recovery_count = 0
            self.return_history = []
