"""
The training pipeline for RL models.

This module coordinates the complete training process:
- setting random seeds for reproducibility
- loading and inspecting market data
- creating training, validation, and test environments
- initializing PPO or SAC agents
- evaluating models during training
- saving trained models and running post-training diagnostics
"""
import random
import numpy as np
import torch

from stable_baselines3 import PPO, SAC

from data.pipeline import load_data
from train.make_env import make_envs
from train.utils.inspect_data import inspect_observation
from train.utils.run_info import run_debugging_info

seeds = [0, 1, 2, 3, 4]
folds = [
    ("2012-12-31", "2013-01-01", "2014-12-31"),
    ("2014-12-31", "2015-01-01", "2016-12-31"),
    ("2016-12-31", "2017-01-01", "2018-12-31"),
    ("2018-12-31", "2019-01-01", "2020-12-31"),
]

def run_training(rl_algorithm="PPO"):
    """
        Train RL agents for portfolio allocation.

        The function trains multiple agents using different random seeds to
        evaluate robustness. For each seed, it creates independent environments,
        trains either a PPO/SAC model, evaluates performance on a validation
        environment, saves the final model, and runs debugging analysis on the
        test environment.

        rl_algorithm : str
            RL algorithm to use. Supported values are:
            - "PPO" for Proximal Policy Optimization
            - "SAC" for Soft Actor-Critic
        """
    for fold_idx, (train_end, val_start, val_end) in enumerate(folds):
        # Seeds for reproducibility
        for seed in seeds:
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
            ) = load_data()
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
                total_timesteps=200_000,
            )

            # best_model saved automatically during training based on val performance
            # final_model is the state after the last update
            model.save(f"../models/final_model_seed_{seed}")

            run_debugging_info(
                model, test_env, test_returns, seed, PPO if isinstance(model, PPO) else SAC
            )

if __name__ == "__main__":
    run_training()
