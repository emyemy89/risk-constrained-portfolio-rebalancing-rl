from stable_baselines3 import PPO
from train.make_env import make_envs

# Data Loading

train_env = make_envs()

model = PPO("MlpPolicy", train_env)