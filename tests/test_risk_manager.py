"""
Test the risk manager from /risk/risk_manager.py
"""
from risk.risk_manager import RiskManager

def test_stops_when_drawdown_reaches_threshold():
    """Stop trading when portfolio drawdown reaches the configured threshold."""
    risk = RiskManager(max_drawdown=0.08, cooldown_days=5)

    risk.update(100.0, was_stopped=False)
    risk.update(95.0, was_stopped=False)
    risk.update(91.0, was_stopped=False)

    assert risk.stopped is True
    assert risk.stop_reason == "drawdown"
    assert risk.cooldown_remaining == 5


def test_does_not_stop_before_drawdown_threshold():
    """Continue trading while portfolio drawdown remains below the threshold."""
    risk = RiskManager(max_drawdown=0.08)

    risk.update(100.0, was_stopped=False)
    risk.update(95.0, was_stopped=False)
    risk.update(93.0, was_stopped=False)

    assert risk.stopped is False


def test_cooldown_resumes_trading_after_configured_days():
    """Resume trading after the configured cooldown period has elapsed."""
    risk = RiskManager(cooldown_days=3)

    risk.update(100.0, was_stopped=False)
    risk.update(90.0, was_stopped=False)

    assert risk.stopped is True

    risk.update(90.0, was_stopped=True)
    assert risk.stopped is True

    risk.update(90.0, was_stopped=True)
    assert risk.stopped is True

    risk.update(90.0, was_stopped=True)
    assert risk.stopped is False
    assert risk.cooldown_remaining == 0