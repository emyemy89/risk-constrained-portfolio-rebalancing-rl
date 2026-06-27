import numpy as np
import pandas as pd
from aligner import align_assets

def validate_data(data):
    if data.empty:
        raise ValueError("Input data is empty")
    if data.index.duplicated().sum() > 0:
        raise ValueError("Duplicate index found")
    return data

def create_features(data):
    close_prices = data["Close"]
    # We need to align so all 4 ETF:s start at the same time to match future matrix
    # For the moment, we work with only one column
    aligned_prices = align_assets(close_prices)
    # Using r_t = log(P_t / P_{t-1})
    # Remove the first NaN raw
    log_returns = np.log(aligned_prices / aligned_prices.shift(1)).dropna()
    # Seeing only today's return is not relevant enough
    # We also care to see how turbulent has the asset been recently
    # we compute the std_dev or returns σ_t=std(r_t−19,...,r_t), 20 is approx. 1 month of trades
    volatility_20 = log_returns.rolling(20).std() # volatility = 0.009785 ==> 0.97%
    # We add momentum to observe the general direction, up or down
    # (momentum is supposed to be smaller than volatility, because avg_return << std_dev)
    momentum_20 = log_returns.rolling(20).mean()
    # Combine the features
    features = pd.concat(
        [
            log_returns,
            volatility_20,
            momentum_20
        ],
        axis=1,
        keys=["ret", "vol", "mom"]
    ).dropna()
    return features
