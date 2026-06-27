# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from load import *
from aligner import *
from features.windowing import create_windows

# %% [markdown]
# # 2. Validate
# %%
# Missing values
data.isna().sum()
# %%
data.index.duplicated().sum()
print(data.shape)
print(data.head())
print(data.columns)
# %%
close_prices = data["Close"]

for ticker in close_prices.columns:
    print(
        ticker,
        close_prices[ticker].first_valid_index(),
        close_prices[ticker].last_valid_index()
    )
# %% [markdown]
# # 3. Align assets
# %%
# We need to align so all 4 ETF:s start at the same time to match future matrix
# For the moment, we work with only one column
close_prices = data["Close"]

aligned_prices = align_assets(close_prices)
print(aligned_prices.head())
print(aligned_prices.shape)
aligned_prices.isna().sum().sum()
# %% [markdown]
# # 4. Convert raw to log returns
# %%
# Using r_t = log(P_t / P_{t-1})
log_returns = np.log(aligned_prices / aligned_prices.shift(1))
# %%
# Remove the first NaN raw
log_returns = log_returns.dropna()
print(log_returns.shape)
log_returns.describe()
# %% [markdown]
# # 5. Create rolling observation window
# %%
# Seeing only today's return is not relevant enough
# We also care to see how turbulent has the asset been recently
# we compute the std_dev or returns σ_t=std(r_t−19,...,r_t), 20 is approx. 1 month of trades
volatility_20 = log_returns.rolling(20).std()
print(volatility_20.head(25)) # volatility = 0.009785 ==> 0.97%
# %%
# We add momentum to observe the general direction, up or down
# (momentum is supposed to be smaller than volatility, because avg_return << std_dev)
momentum_20 = log_returns.rolling(20).mean()
print(momentum_20.head(25))
# %% [markdown]
# # 6. Generate derived features
# %%
# Combine the features
features = pd.concat(
    [
        log_returns,
        volatility_20,
        momentum_20
    ],
    axis=1,
    keys=["ret", "vol", "mom"]
)
# Drop the first 19 NaN
features = features.dropna()

print(features.shape)
print(features.head())
# %% [markdown]
# # 7. Create Splits
# %%
# Train, Validation, Test
train_features = features.loc[:'2018-12-31']
val_features = features.loc['2019-01-01':'2021-12-31']
test_features = features.loc['2022-01-01':]

print("Train:", train_features.shape)
print("Validation:", val_features.shape)
print("Test:", test_features.shape)

print(train_features.index.min(), train_features.index.max())
print(val_features.index.min(), val_features.index.max())
print(test_features.index.min(), test_features.index.max())
# %% [markdown]
# # 8. Normalize
# %%
# Use z-score standardization
scaler = StandardScaler()
scaler.fit(train_features)

# Transform each split
train_scaled = pd.DataFrame(
    scaler.transform(train_features),
    index=train_features.index,
    columns=train_features.columns,
)

val_scaled = pd.DataFrame(
    scaler.transform(val_features),
    index=val_features.index,
    columns=val_features.columns,
)

test_scaled = pd.DataFrame(
    scaler.transform(test_features),
    index=test_features.index,
    columns=test_features.columns,
)

train_scaled.describe() # mean is approx 0 and std_dev approx 1
# %%
# Convert time series to window observation
WINDOW_SIZE = 30

train_windows, train_dates = create_windows(
    train_scaled,
    WINDOW_SIZE,
)
val_windows, val_dates = create_windows(
    val_scaled,
    WINDOW_SIZE,
)
test_windows, test_dates = create_windows(
    test_scaled,
    WINDOW_SIZE,
)
# %%
