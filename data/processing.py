import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def align_assets(prices):
    return prices.dropna()

def validate_data(data):
    if data.empty:
        raise ValueError("Input data is empty")
    if data.index.duplicated().sum() > 0:
        raise ValueError("Duplicate index found")
    return data

def compute_log_returns(aligned_prices):
    # Using r_t = log(P_t / P_{t-1})
    # Remove the first NaN raw
    log_returns = np.log(aligned_prices / aligned_prices.shift(1)).dropna()
    return log_returns

def compute_volatility(log_returns, time_interval):
    """
     Seeing only today's return is not relevant enough
     We also care to see how turbulent has the asset been recently
     we compute the std_dev or returns σ_t=std(r_t−19,...,r_t), 20 is approx. 1 month of trades
    """
    volatility = log_returns.rolling(time_interval).std()  # volatility = 0.009785 ==> 0.97%
    return volatility

def compute_momentum(log_returns, time_interval):
    # We add momentum to observe the general direction, up or down
    # (momentum is supposed to be smaller than volatility, because avg_return << std_dev)
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


def create_features(data):
    """
    Define features used in RL training.
    Currently it holds 19 features: 5 ret, 5 vol, 5 mom, 5 corr
    :param data:
    :return:
    """
    close_prices = data["Close"]
    # We need to align so all 4 ETF:s start at the same time to match future matrix
    # For the moment, we work with only one column
    aligned_prices = align_assets(close_prices)
    # get the features
    log_returns = compute_log_returns(aligned_prices)
    volatility_20 = compute_volatility(log_returns, 20)
    momentum_20 = compute_momentum(log_returns, 20)
    correlations_20 = compute_correlation(log_returns, 20)

    # Combine the features
    features = pd.concat(
        [
            log_returns,
            volatility_20,
            momentum_20,
            correlations_20,
        ],
        axis=1,
        keys=["ret", "vol", "mom", "corr"],
    ).dropna()
    return features, log_returns.loc[features.index]

def split_data(features):
    return( features.loc[:'2018-12-31'],
            features.loc['2019-01-01':'2021-12-31'],
            features.loc['2022-01-01':])

def scale_features(features):
    scaler = StandardScaler()
    scaler.fit(features)
    return  pd.DataFrame(
        scaler.transform(features),
        index=features.index,
        columns=features.columns,
    )