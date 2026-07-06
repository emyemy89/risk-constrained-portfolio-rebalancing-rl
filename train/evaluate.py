import numpy as np

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
