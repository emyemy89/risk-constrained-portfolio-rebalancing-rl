import random
import numpy as np
import torch

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback

from data.pipeline import load_training_data
from train.make_env import make_envs
from train.utils.inspect_data import inspect_observation
from train.utils.run_info import run_debugging_info

def run_training(rl_algorithm="SAC"):
    # Seeds for reproducibility
    SEEDS = [0, 1, 2, 3, 4]
    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
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


        # create the environments
        train_env, val_env, test_env = make_envs(
            train_windows,
            train_returns,
            val_windows,
            val_returns,
            test_windows,
            test_returns,
        )
        train_env.reset(seed=seed)
        val_env.reset(seed=seed)

        # Evaluation callback
        # pauses training -> runs policy on val env -> computes avg performance -> saves best model(if improved)
        eval_callback = EvalCallback(
            val_env,
            best_model_save_path=f"../models/best_model_seed_{rl_algorithm}_{seed}/",
            log_path=f"../logs/seed_{rl_algorithm}_{seed}/",
            eval_freq=10_000, #evaluates every 10K environment steps
            deterministic=True,
            render=False,
        )
        if rl_algorithm == "PPO":
            model = PPO(
                policy="MlpPolicy",
                env=train_env,
                seed=seed,
                learning_rate=1e-4,
                n_steps=2048,  # rollout length
                batch_size=64,  # mini-batch size
                n_epochs=10,  # how many times PPO reuses the collected rollout
                gamma=0.99,  # long-term reward discount (how much agent values future rewards)
                gae_lambda=0.95,  # advantage smoothing(how are they estimated)
                clip_range=0.2,  # what makes PPO "proximal" and stable.
                target_kl=0.02,
                ent_coef=0.01,  # exploration vs. value learning balance (vf_coef)
                vf_coef=0.5,
                tensorboard_log="../logs/tensorboard/",  # logs
                verbose=0,
            )
        else:
            model = SAC(
                policy="MlpPolicy",
                env=train_env,
                seed=seed,
                learning_rate=3e-4,
                buffer_size=100_000,
                learning_starts=1_000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                ent_coef="auto",
                tensorboard_log="../logs/tensorboard/", # logs
                verbose=0,
            )

        model.learn(
            total_timesteps=10_000,
            callback=eval_callback,
        )

        # best_model saved automatically during training based on val performance
        # final_model is the state after the last update
        model.save(f"../models/final_model_seed_{rl_algorithm}_{seed}")

        model_class = PPO if rl_algorithm == "PPO" else SAC
        run_debugging_info(model, test_env, test_returns, seed, model_class, rl_algorithm)

if __name__ == "__main__":
    run_training()

