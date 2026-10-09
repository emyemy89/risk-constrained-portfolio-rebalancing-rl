


class RiskManager:
    def __init__(self, max_consecutive_losses=3, recovery_days=3):
        self.max_consecutive_losses = max_consecutive_losses
        self.recovery_days = recovery_days
        self.consecutive_losses = 0
        self.recovery_count = 0
        self.stopped = False


    def update(self, daily_return, market_recovered=False):
        if not self.stopped:
            if daily_return < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0

            if self.consecutive_losses >= self.max_consecutive_losses:
                self.stopped = True
                self.recovery_count = 0

        else:
            if market_recovered:
                self.recovery_count += 1
            else:
                self.recovery_count = 0

            if self.recovery_count >= self.recovery_days:
                self.stopped = False
                self.consecutive_losses = 0
                self.recovery_count = 0

        return self.stopped
