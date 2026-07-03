from stable_baselines3 import PPO
from train.make_env import make_envs

# Data Loading

train_env, val_env, test_env = make_envs()

model = PPO(
     "MlpPolicy",
            env=train_env,
            verbose=1,
            n_steps=2048, # rollout length
            batch_size=64, # minibatch size
            gamma=0.99, # long term reward discount
            gae_lambda=0.95 # advantage smoothing
    )
