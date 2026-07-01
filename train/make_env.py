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
