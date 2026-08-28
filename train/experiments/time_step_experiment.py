import matplotlib.pyplot as plt
import pandas as pd

from train.run_training import run_training


timesteps = [10_000,50_000,100_000,150_000,200_000,300_000,500_000,800_000,2_000_000,10_000_000]

results = []

for total_timesteps in timesteps:
    print(f"\n===== {total_timesteps} timesteps =====")

    metrics = run_training(
        rl_algorithm="PPO",
        total_timesteps=total_timesteps,
    )

    results.append({
        "timesteps": total_timesteps,
        "Annual Return": metrics["Annual Return"],
        "Annual Volatility": metrics["Annual Volatility"],
        "Sharpe Ratio": metrics["Sharpe Ratio"],
        "Max Drawdown": metrics["Max Drawdown"],
    })


results_df = pd.DataFrame(results)

print("\n===== Results =====")
print(results_df)


# Annual return
plt.figure()
plt.plot(results_df["timesteps"], results_df["Annual Return"], marker="o")
plt.xscale("log")
plt.xlabel("Total Timesteps")
plt.ylabel("Annual Return")
plt.title("Annual Return vs Training Timesteps")
plt.grid()
plt.show()


# Annual volatility
plt.figure()
plt.plot(results_df["timesteps"], results_df["Annual Volatility"], marker="o")
plt.xscale("log")
plt.xlabel("Total Timesteps")
plt.ylabel("Annual Volatility")
plt.title("Annual Volatility vs Training Timesteps")
plt.grid()
plt.show()


# Sharpe ratio
plt.figure()
plt.plot(results_df["timesteps"], results_df["Sharpe Ratio"], marker="o")
plt.xscale("log")
plt.xlabel("Total Timesteps")
plt.ylabel("Sharpe Ratio")
plt.title("Sharpe Ratio vs Training Timesteps")
plt.grid()
plt.show()


# Maximum drawdown
plt.figure()
plt.plot(results_df["timesteps"], results_df["Max Drawdown"], marker="o")
plt.xscale("log")
plt.xlabel("Total Timesteps")
plt.ylabel("Maximum Drawdown")
plt.title("Maximum Drawdown vs Training Timesteps")
plt.grid()
plt.show()