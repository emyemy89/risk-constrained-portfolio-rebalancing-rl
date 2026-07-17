import matplotlib.pyplot as plt

def plot_portfolios(strategy_values):
    plt.figure(figsize=(10, 5))
    for name, values in strategy_values.items():
        plt.plot(values, label=name)
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value")
    plt.title("Portfolio Performance Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_weights(weights):
    plt.figure(figsize=(10,5))
    plt.plot(weights)
    plt.title("PPO Portfolio Weights")
    plt.xlabel("Trading Days")
    plt.ylabel("Weight")
    plt.legend(["SPY", "QQQ", "TLT", "GLD", "VNQ", "CASH"])
    plt.grid(True)
    plt.show()


def inspect_observation(window):
    """
    Sanity check to ensure observation looks fine. Should be:
        Returns: noisy, oscillating around 0
        volatility: smoother, slowly changing.
        Momentum: smoother than returns, showing trends.
    :param window:
    :return: None
    """
    print("Shape:", window.shape)
    plt.figure(figsize=(12, 6))
    plt.plot(window)
    plt.title("Observation Window")
    plt.xlabel("Past Trading Days")
    plt.ylabel("Standardized Feature Value")
    plt.legend([
        "SPY_ret", "QQQ_ret", "TLT_ret", "GLD_ret", "VNQ_ret",
        "SPY_vol", "QQQ_vol", "TLT_vol", "GLD_vol", "VNQ_vol",
        "SPY_mom", "QQQ_mom", "TLT_mom", "GLD_mom", "VNQ_mom",
    ])
    plt.grid(True)
    plt.show()