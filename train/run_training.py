import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from data.pipeline import load_training_data
from train.make_env import make_envs
from train.evaluate import inspect_weights, evaluate_portfolio, evaluate_baseline, compute_metrics
from train.inspect_data import inspect_observation,  plot_portfolios, plot_weights

def run_training():
    # Data Loading
    (
        train_windows,
        train_returns,
        val_windows,
        val_returns,
        test_windows,
        test_returns,
    ) = load_training_data()
    inspect_observation(train_windows[0])


    # spy = np.array([1, 0, 0, 0, 0])
    # equal = np.ones(5) / 5
    # print("SPY:", evaluate_baseline(test_returns, spy))
    # print("Equal weight:", evaluate_baseline(test_returns, equal))

    train_env, val_env, test_env = make_envs(
        train_windows,
        train_returns,
        val_windows,
        val_returns,
        test_windows,
        test_returns,
    )

    # Evaluation callback
    # pauses training -> runs policy on val env -> computes avg performance -> saves best model(if improved)
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path="../models/", # saves the checkpoint with the best validation performance
        log_path="../logs/",
        eval_freq=10_000, #evaluates every 10K environment steps
        deterministic=True,
        render=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=1e-4,
        n_steps=2048, # rollout length
        batch_size=64, # mini-batch size
        n_epochs=10, # how many times PPO reuses the collected rollout
        gamma=0.99, # long-term reward discount (how much agent values future rewards)
        gae_lambda=0.95, # advantage smoothing(how are they estimated)
        clip_range=0.2, # what makes PPO "proximal" and stable.
        target_kl=0.02,
        ent_coef=0.0, # exploration vs. value learning balance (vf_coef)
        vf_coef=0.5,
        tensorboard_log="../logs/tensorboard/", # logs
        verbose=1,
    )

    model.learn(
        total_timesteps=200_000,
        callback=eval_callback,
    )

    # best_model saved automatically during training based on val performance
    # final_model is the state after the last update
    model.save("../models/final_model")

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

    spy = evaluate_baseline(test_returns, np.array([1, 0, 0, 0, 0]))
    equal = evaluate_baseline(test_returns, np.ones(5) / 5)
    print("PPO:", metrics)
    print("SPY:", spy)
    print("Equal:", equal)



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


if __name__ == "__main__":
    run_training()

