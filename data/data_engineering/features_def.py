"""Feature engineering utilities for portfolio.

This module transforms aligned asset price data into model-ready features,
including log returns, volatility, momentum, rolling correlations, trend
regimes, and drawdown indicators.
"""

import numpy as np
import pandas as pd

from data.data_engineering.processing import align_assets

def create_features(data, rolling_window):
    """
    Combines all features used in RL training
    Currently it holds 19 features: 5 ret, 5 vol, 5 mom, 5 corr
    :param data: the ETF data being used
    :return: features, log_returns
    """
    close_prices = data["Close"]

    # We need to align so all 4 ETF:s start at the same time to match future matrix
    # For the moment, we work with only one column
    aligned_prices = align_assets(close_prices)

    # get the features
    log_returns = compute_log_returns(aligned_prices)
    log_returns["CASH"] = 0.0 # Agent will se it as an asset with no return, vol or mom

    volatility_20 = compute_volatility(log_returns, rolling_window)

    momentum_20 = compute_momentum(log_returns, rolling_window)

    correlations = compute_correlation(log_returns.drop(columns=["CASH"]), rolling_window)
    correlations["CASH_corr_SPY"] = 0.0
    spy_ma50 = compute_trend_regime(50, aligned_prices)
    spy_ma200 = compute_trend_regime(200, aligned_prices)
    spy_drawdown = compute_drawdown(252, aligned_prices)


    # Combine the features
    features = pd.concat(
        [
            log_returns,
            volatility_20,
            momentum_20,
            correlations, spy_ma50, spy_ma200,
            spy_drawdown,
        ],axis=1, keys=["ret", "vol20", "mom20",
                        "corr", "spy_ma50", "spy_ma200", "spy_drawdown",]
    ).dropna()
    return features, log_returns.loc[features.index]

def compute_log_returns(aligned_prices):
    """
    Compute log returns of each asset in aligned_prices
    Using r_t = log(P_t / P_{t-1})
    Remove the first NaN raw
    """
    log_returns = np.log(aligned_prices / aligned_prices.shift(1)).dropna()
    return log_returns

def compute_volatility(log_returns, time_interval):
    """
    Compute volatility of each asset in aligned_prices
    Seeing only today's return is not relevant enough
    We also care to see how turbulent has the asset been recently
    we compute the std_dev or returns σ_t=std(r_t−19,...,r_t), 20 is approx. 1 month of trades
    """
    volatility = log_returns.rolling(time_interval).std()  # volatility = 0.009785 ==> 0.97%
    return volatility

def compute_momentum(log_returns, time_interval):
    """
    We add momentum to observe the general direction, up or down
    (momentum is supposed to be smaller than volatility, because avg_return << std_dev)
    """
    momentum = log_returns.rolling(time_interval).mean()
    return momentum

def compute_correlation(log_returns, time_interval):
    """
        Compute rolling correlation of each asset with SPY
        :param log_returns: DataFrame of aligned asset log returns
                            columns = ETF tickers
        :param window: rolling correlation window
        :return: DataFrame containing rolling correlations with SPY
    """
    spy_returns = log_returns["SPY"]
    correlations = pd.DataFrame(index=log_returns.index)
    for asset in log_returns.columns:
        if asset != "SPY":
            correlations[f"{asset}_corr_SPY"] = (
                log_returns[asset]
                .rolling(time_interval)
                .corr(spy_returns)
            )
    return correlations

def compute_trend_regime(window_size, aligned_prices):
    """
    Compute trend regime using X-day moving average
    """
    spy = aligned_prices["SPY"]
    spy_ma = spy.rolling(window_size).mean() # moving average
    spy_ma_ratio = spy / spy_ma # 1 → SPY above its 50-day trend<1 → SPY below its 50-day trend
    return spy_ma_ratio

def compute_drawdown(window_size, aligned_prices):
    """
    Compute drawdown using X-day moving average
    """
    spy = aligned_prices["SPY"]
    rolling_max = spy.rolling(window_size).max()
    spy_drawdown = spy / rolling_max - 1
    return spy_drawdown

def compute_market_volatility(volatility):
    """
    Compute market volatility of non-CASH assets
    """
    return volatility.drop(columns=["CASH"]).mean(axis=1).rename("market_volatility")

def compute_market_correlations(correlations):
    """
    Compute average correlation with SPY across non-CASH assets.
    """
    return correlations[
        [column for column in correlations.columns if column != "CASH_corr_SPY"]
    ].mean(axis=1).rename("market_correlation")
