"""
This module encapsulated procedures for choosing which Reinforcement Learning algorithm to use
"""
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO

def create_model(rl_algorithm, train_env, seed=None):
    """
    Choose between PPO and SAC
    """
    if rl_algorithm == "PPO":
        return RecurrentPPO(
            policy="MlpLstmPolicy",
            env=train_env,
            seed=seed,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            target_kl=0.02,
            ent_coef=0.01,
            vf_coef=0.5,
            tensorboard_log="../logs/tensorboard/",
            verbose=0,
        )

    return SAC(
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
        tensorboard_log="../logs/tensorboard/",
        verbose=0,
    )