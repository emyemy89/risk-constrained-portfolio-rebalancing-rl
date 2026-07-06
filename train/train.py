from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from data.pipeline import load_training_data
from train.make_env import make_envs

# Data Loading
(
    train_windows,
    train_returns,
    val_windows,
    val_returns,
    test_windows,
    test_returns,
) = load_training_data()

train_env, val_env, test_env = make_envs(
    train_windows,
    train_returns,
    val_windows,
    val_returns,
    test_windows,
    test_returns,
)
train_env, val_env, test_env = make_envs()

# Evaluation callback
# pauses training -> runs policy on val env -> computes avg performance -> saves best model(if improved)
eval_callback = EvalCallback(
    val_env,
    best_model_save_path="./models/",
    log_path="./logs/",
    eval_freq=10_000,
    deterministic=True,
    render=False,
)

model = PPO(
     "MlpPolicy",
            env=train_env,
            verbose=1,
            n_steps=2048, # rollout length
            batch_size=64, # minibatch size
            gamma=0.99, # long term reward discount
            gae_lambda=0.95 # advantage smoothing
    )

model.learn(
    total_timesteps=200_000,
    callback=eval_callback,
)

model.save("final_model")

