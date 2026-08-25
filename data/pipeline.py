"""
Training data pipeline orchestration.

This module combines data loading, validation and  feature engineering as well as
time-series window generation into a single pipeline used to prepare observations
and returns for training.
"""
from data.extract.load_data import load_etf_data
from data.data_engineering.features_def import create_features
from data.data_engineering.processing import (
             validate_data, split_data, split_test_data, scale_features)
from features.windowing import create_windows


def load_data(train_end, val_start, val_end,
              obs_window_size = 30, rolling_window = 20):
    """
    Prepare training, validation, and test datasets for training.

    The pipeline: (1)loads asset price data, (2)creates financial features,
    (3)splits data chronologically, (4)scales features using the training set,
    and (5)converts time-series data into rolling observation windows.

    :return: Tuple containing train/validation/test observation windows
             and corresponding asset returns.
    """
    # (1) Load and validate
    data = validate_data(load_etf_data())

    #(2) Create financial features
    # Align assets -> Convert raw to log returns -> Calculate volatility and momentum -> Concatenate
    features, returns = create_features(data,rolling_window)

    #(3) Create Splits
    train_features, val_features = split_data(features,
        train_end=train_end, val_start=val_start, val_end=val_end)
    train_returns, val_returns = split_data(returns,
        train_end=train_end, val_start=val_start, val_end=val_end)


    # Normalize using z-score standardization fit on train set only
    train_scaled, val_scaled = scale_features(train_features, val_features)

    # Convert time series to window observation
    train_windows, train_dates = create_windows(train_scaled, obs_window_size)
    val_windows, val_dates = create_windows(val_scaled, obs_window_size)

    # return must match the observation, so we shift
    offset = obs_window_size - 1

    train_returns = train_returns.iloc[offset:].to_numpy()
    val_returns = val_returns.iloc[offset:].to_numpy()

    return (
        train_windows, train_returns,
        val_windows, val_returns,
        train_features.columns,
    )

def load_test_data(train_end="2021-12-31", test_start="2022-01-01",
    obs_window_size=30, rolling_window=20,):
    """Prepare final training and untouched test datasets."""

    data = validate_data(load_etf_data())
    features, returns = create_features(data, rolling_window)

    # (2) Split
    train_features, test_features = split_test_data(features, train_end, test_start)
    train_returns, test_returns = split_test_data(returns, train_end, test_start)

    # (3) Fit scaler ONLY on pre-test data
    train_scaled, test_scaled = scale_features(train_features, test_features)


    train_windows, train_dates = create_windows(train_scaled, obs_window_size)
    test_windows, test_dates = create_windows(test_scaled, obs_window_size)

    # return must match the observation, so we shift
    offset = obs_window_size - 1
    train_returns = train_returns.iloc[offset:].to_numpy()
    test_returns = test_returns.iloc[offset:].to_numpy()

    return (
        train_windows, train_returns,
        test_windows, test_returns,)
