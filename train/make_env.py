"""
Utilities for creating and wrapping portfolio environments.

This module provides helper functions for initializing PortfolioEnv instances
and wrapping them with Stable-Baselines3 Monitor objects.
"""
from stable_baselines3.common.monitor import Monitor
from env.portfolio_env import PortfolioEnv

def make_env(windows, returns, valuation_signal, **kwargs):
    """
    Create a PortfolioEnv instance
    Additional keyword arguments are forwarded to PortfolioEnv
    """
    env = PortfolioEnv(
        windows=windows,
        returns=returns,
        valuation_signal=valuation_signal,
        **kwargs,
    )
    return Monitor(env)

def make_envs(
    train_windows,
    train_returns,
    val_windows,
    val_returns,
    test_windows,
    test_returns,
    train_valuation_signal=None,
    val_valuation_signal=None,
    test_valuation_signal=None,
    **kwargs,
):
    """
        Create training, validation, and testing environments.

        Generates separate environments using different datasets while
        sharing the same environment configuration.
        """
    train_env = make_env(train_windows, train_returns, train_valuation_signal, **kwargs)
    val_env = make_env(val_windows, val_returns, val_valuation_signal, **kwargs)
    test_env = make_env(test_windows, test_returns, test_valuation_signal, **kwargs)
    return train_env, val_env, test_env
