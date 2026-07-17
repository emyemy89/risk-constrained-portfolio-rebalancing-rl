from stable_baselines3 import PPO

from train.evaluate import *
from train.inspect_data import  plot_portfolios, plot_weights


def run_debugging_info(model, test_env, test_returns):
    inspect_weights(
        model,
        test_env
    )
    best_model = PPO.load("../models/best_model", env=test_env)
    results = evaluate_portfolio(
        best_model,
        test_env,
    )
    print(results["portfolio_values"][-1])
    total_return = results["portfolio_values"][-1] - 1
    print("Test cumulative return:", total_return)

    # compute metrics eval
    metrics = compute_metrics(
        results["portfolio_values"],
        results["daily_returns"],
    )
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    spy = evaluate_baseline(test_returns, np.array([1, 0, 0, 0, 0, 0]))
    equal = evaluate_baseline(test_returns, np.ones(6) / 6)
    print("PPO:", metrics)
    print("SPY:", spy)
    print("Equal:", equal)

    weight_statistics(results["weights"])
    turnover_statistics(results["weights"])
    equal_weight_distance(results["weights"])

    # plot portfolio values
    ppo_values = results["portfolio_values"]
    spy_returns = test_returns @ np.array([1, 0, 0, 0, 0])
    equal_returns = test_returns @ (np.ones(5) / 5)
    spy_values = np.exp(np.cumsum(spy_returns))
    equal_values = np.exp(np.cumsum(equal_returns))
    plot_portfolios({
        "PPO": ppo_values,
        "SPY": spy_values,
        "Equal Weight": equal_values,
    })

    # plot weights
    plot_weights(results["weights"])