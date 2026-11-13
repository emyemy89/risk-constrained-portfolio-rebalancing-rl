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
                 max_rolling_loss=0.05, max_drawdown=0.08,
                 cooldown_days=5, retrigger_buffer=0.05):
        self.max_consecutive_losses = max_consecutive_losses
        self.rolling_window = rolling_window
        self.max_rolling_loss = max_rolling_loss
        self.max_drawdown = max_drawdown
        self.recovery_days = recovery_days
        self.cooldown_days = cooldown_days
        self.retrigger_buffer = retrigger_buffer
        self.reset()

    def reset(self):
        self.consecutive_losses = 0
        self.recovery_count = 0
        self.return_history = []
        self.peak_value = None
        self.stopped = False
        self.stop_reason = None
        self.cooldown_remaining = 0
        self.resume_drawdown = None
        self.previous_drawdown = 0.0

    def update(self, portfolio_value, was_stopped):
        if self.peak_value is None:
            self.peak_value = portfolio_value

        self.peak_value = max(self.peak_value, portfolio_value)
        drawdown = portfolio_value / self.peak_value - 1.0

        # Continue the fixed cash cooldown.
        if was_stopped:
            self.cooldown_remaining -= 1

            if self.cooldown_remaining <= 0:
                self.stopped = False
                self.stop_reason = None
                self.resume_drawdown = drawdown
                self.previous_drawdown = drawdown

            return

        # A recovery above the drawdown threshold re-arms the normal trigger.
        if drawdown > -self.max_drawdown:
            self.resume_drawdown = None

        if self.resume_drawdown is None:
            should_stop = (
                    self.previous_drawdown > -self.max_drawdown
                    and drawdown <= -self.max_drawdown
            )
        else:
            should_stop = (
                    drawdown <= self.resume_drawdown - self.retrigger_buffer
            )

        if should_stop:
            self.stopped = True
            self.stop_reason = "drawdown"
            self.cooldown_remaining = self.cooldown_days

        self.previous_drawdown = drawdown
