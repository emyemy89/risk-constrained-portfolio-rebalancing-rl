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
import pandas as pd
import torch

from data.pipeline import load_data, load_test_data
from train.make_env import make_env
from train.utils.inspect_data import inspect_observation
from train.utils.run_info import run_debugging_info
from train.utils.algorithm_selection import create_model
from train.evaluate import evaluate_and_compute_metrics, print_validation_results

seeds = [0, 1, 2, 3, 4]
folds = [
    ("2012-12-31", "2013-01-01", "2014-12-31"),
    ("2014-12-31", "2015-01-01", "2016-12-31"),
    ("2016-12-31", "2017-01-01", "2018-12-31"),
    ("2018-12-31", "2019-01-01", "2020-12-31"),]

def run_training(rl_algorithm="PPO", total_timesteps=50_000):
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
    validation_results = []
    for fold_idx, (train_end, val_start, val_end) in enumerate(folds):
        # Seeds for reproducibility
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            # Data Loading
            (train_windows, train_returns, val_windows, val_returns, feature_columns) = (
                load_data(train_end=train_end, val_start=val_start, val_end=val_end,))

            inspect_observation(train_windows[0], feature_columns)


            # create the environments
            train_env = make_env(train_windows, train_returns)
            val_env = make_env(val_windows, val_returns)

            train_env.reset(seed=seed)
            val_env.reset(seed=seed)

            model = create_model(rl_algorithm, train_env, seed,)
            model.learn(total_timesteps=total_timesteps)

            # best_model saved automatically during training based on val performance
            # final_model is the state after the last update
            #model.save(f"../models/final_model_seed_{seed}")

            metrics = evaluate_and_compute_metrics(model, val_env)
            validation_results.append({"fold": fold_idx + 1,"seed": seed, **metrics,})
            run_debugging_info(model, val_env, val_returns, seed=seed,fold_idx=fold_idx + 1,)
    results_df = pd.DataFrame(validation_results)

    # Print validation results
    print_validation_results(results_df)

    # Final training on all data up to 2021
    (train_windows, train_returns, test_windows,test_returns,
    ) = load_test_data()

    train_env = make_env(train_windows, train_returns)
    test_env = make_env(test_windows, test_returns)

    train_env.reset(seed=0)
    test_env.reset(seed=0)

    model = create_model(rl_algorithm, train_env, seed=0)

    model.learn(total_timesteps=total_timesteps)
    model.save("../models/final_model")

    final_metrics = run_debugging_info(model, test_env, test_returns, seed=0, fold_idx="final")
    return final_metrics

if __name__ == "__main__":
    run_training()
