import matplotlib.pyplot as plt


def inspect_observation(window):
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