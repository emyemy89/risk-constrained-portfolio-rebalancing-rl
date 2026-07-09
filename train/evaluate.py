import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(model, env, num_episodes=5):
    """
    Runs deterministic eval, returns mean episode return
    """
    episode_returns = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_return = 0
        while not done:
            action = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_return += reward
            done = truncated or terminated
        episode_returns.append(total_return)
    return float(np.mean(episode_returns))

def inspect_weights(model, env, n_steps=10):
    obs, _ = env.reset()
    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            "weights:",
            np.round(info["weights"], 3),
            "reward:",
            round(reward, 4)
        )
        if terminated or truncated:
            break

def evaluate_portfolio(model, env):
    obs, _ = env.reset()
    portfolio_values = []
    weights_history = []
    rewards = []
    daily_returns = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
        weights_history.append(info["weights"])
        rewards.append(reward)
        #daily_returns.append(info["cumm_return"])
        daily_returns.append(info["step_return"])
        done = terminated or truncated
    return {
        "portfolio_values": np.array(portfolio_values),
        "weights": np.array(weights_history),
        "rewards": np.array(rewards),
        "daily_returns": np.array(daily_returns),
    }

def evaluate_baseline(returns, weights):
    step_returns = returns @ weights
    portfolio_values = np.exp(np.cumsum(step_returns))
    metrics = compute_metrics(
        portfolio_values,
        step_returns,
    )
    return metrics

def compute_metrics(portfolio_values, daily_returns, risk_free_rate=0.0):
    annual_return = portfolio_values[-1] ** (252 / len(daily_returns)) - 1
    annual_vol = np.std(daily_returns) * np.sqrt(252)
    sharpe = (
        (annual_return - risk_free_rate) / annual_vol
        if annual_vol > 0 else 0.0
    )
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = portfolio_values / running_max - 1
    max_drawdown = np.min(drawdown)
    return {
        "Annual Return": annual_return,
        "Annual Volatility": annual_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown,
    }

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
