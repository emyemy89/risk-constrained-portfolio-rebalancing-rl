# Pipeline orchestrator
# %%
from data.load_data import load_etf_data
from data.processing import *
from features.windowing import create_windows

# %%
def load_training_data(window_size=30):
    WINDOW_SIZE = 30
    data = load_etf_data()

    # Validate
    data = validate_data(data)

    # %%
    #  Align assets -> Convert raw to log returns -> Calculate volatility and momentum -> Concatenate
    features, returns = create_features(data)
    # # Create Splits
    # Train, Validation, Test
    train_features, val_features, test_features = split_data(features)
    train_returns, val_returns, test_returns = split_data(returns)
    # Normalize using z-score standardization
    train_scaled = scale_features(train_features)
    val_scaled = scale_features(val_features)
    test_scaled = scale_features(test_features) # mean is approx 0 and std_dev approx 1

    # %%
    # Convert time series to window observation
    train_windows, train_dates = create_windows(train_scaled, WINDOW_SIZE)
    val_windows, val_dates = create_windows(val_scaled, WINDOW_SIZE)
    test_windows, test_dates = create_windows(test_scaled, WINDOW_SIZE)

    # return must match the observation, so we shift
    offset = WINDOW_SIZE - 1

    train_returns = train_returns.iloc[offset:].to_numpy()
    val_returns = val_returns.iloc[offset:].to_numpy()
    test_returns = test_returns.iloc[offset:].to_numpy()
    # %%
    return (
        train_windows,
        train_returns,
        val_windows,
        val_returns,
        test_windows,
        test_returns,
    )