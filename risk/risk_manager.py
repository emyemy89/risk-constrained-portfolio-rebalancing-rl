"""
Risk Manager
"""
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

    def record_return(self, daily_return):
        """
        Record the daily return value
        """
        if self.stopped:
            return
        if daily_return < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.stopped = True
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
            self.consecutive_losses = 0
            self.recovery_count = 0

    def reset(self):
        """
        Reset the risk manager
        """
        self.consecutive_losses = 0
        self.recovery_count = 0
        self.stopped = False

