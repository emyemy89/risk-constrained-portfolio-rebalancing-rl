from risk.risk_manager import RiskManager

def test_stops_after_three_consecutive_losses():
    risk = RiskManager(max_consecutive_losses=3)

    assert risk.update(-0.01) is False
    assert risk.update(-0.02) is False
    assert risk.update(-0.01) is True


def test_positive_return_resets_loss_counter():
    risk = RiskManager(max_consecutive_losses=3)

    risk.update(-0.01)
    risk.update(-0.01)
    assert risk.update(0.01) is False
    assert risk.update(-0.01) is False

def test_resumes_after_three_recovery_signals():
    risk = RiskManager(max_consecutive_losses=3, recovery_days=3)

    risk.update(-0.01)
    risk.update(-0.01)
    risk.update(-0.01)
    assert risk.update(0.01, market_recovered=True) is True
    assert risk.update(0.01, market_recovered=True) is True
    assert risk.update(0.01, market_recovered=True) is False
