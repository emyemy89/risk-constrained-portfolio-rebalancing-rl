# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from load_data import load_etf_data
from aligner import *
from processing import *
from features.windowing import create_windows

# %%
data = load_etf_data()
# %% [markdown]
# # 2. Validate
data = data.validate_data()

# %%
# # Align assets -> Convert raw to log returns -> Calculate volatility and momentum -> Concatenate
features = create_features(data)

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
