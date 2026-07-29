from stable_baselines3.common.monitor import Monitor
from env.portfolio_env import PortfolioEnv

def make_env(windows, returns, ranking_probs, **kwargs):
    """
    Create a PortfolioEnv instance
    Additional keyword arguments are forwarded to PortfolioEnv
    """
    env = PortfolioEnv(
        windows=windows,
        returns=returns,
        ranking_probs = ranking_probs,
        **kwargs,
    )
    return Monitor(env)

def make_envs(
    train_windows,
    train_returns,
    train_ranking_probs,
    val_windows,
    val_returns,
    val_ranking_probs,
    test_windows,
    test_returns,
    test_ranking_probs,
    **kwargs,
):
    train_env = make_env(train_windows, train_returns,train_ranking_probs, **kwargs)
    val_env = make_env(val_windows, val_returns, val_ranking_probs, **kwargs)
    test_env = make_env(test_windows, test_returns, test_ranking_probs, **kwargs)

    return train_env, val_env, test_env