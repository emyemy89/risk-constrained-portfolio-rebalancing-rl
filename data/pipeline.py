# Pipeline orchestrator
# %%
from load_data import load_etf_data
from processing import *
from features.windowing import create_windows

# %%
WINDOW_SIZE = 30
data = load_etf_data()

# Validate
data = validate_data(data)

# %%
#  Align assets -> Convert raw to log returns -> Calculate volatility and momentum -> Concatenate
features = create_features(data)
# # Create Splits
# Train, Validation, Test
train_features, val_features, test_features = split_data(features)
# Normalize using z-score standardization
train_scaled = scale_features(train_features)
val_scaled = scale_features(val_features)
test_scaled = scale_features(test_features) # mean is approx 0 and std_dev approx 1

# %%
# Convert time series to window observation
train_windows, train_dates = create_windows(train_scaled, WINDOW_SIZE)
val_windows, val_dates = create_windows(val_scaled, WINDOW_SIZE)
test_windows, test_dates = create_windows(test_scaled, WINDOW_SIZE)
# %%
