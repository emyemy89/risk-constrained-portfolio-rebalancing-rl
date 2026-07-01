from env.portfolio_env import PortfolioEnv

def make_env(windows, returns, **kwargs):
    """
    Create a PortfolioEnv instance
    Additional keyword arguments are forwarded to PortfolioEnv
    """
    return PortfolioEnv(
        windows=windows,
        returns=returns,
        **kwargs,
    )

def make_envs(
    train_windows,
    train_returns,
    val_windows,
    val_returns,
    test_windows,
    test_returns,
    **kwargs,
):
    train_env = make_env(train_windows, train_returns, **kwargs)
    val_env = make_env(val_windows, val_returns, **kwargs)
    test_env = make_env(test_windows, test_returns, **kwargs)

    return train_env, val_env, test_env