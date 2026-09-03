"""
Training data pipeline orchestration.

This module combines data loading, validation and feature engineering as well as
time-series window generation into a single pipeline used to prepare observations
and returns for training, evaluation, and inference.
"""
from data.extract.load_data import load_etf_data
from data.data_engineering.features_def import create_features
from data.data_engineering.processing import (
             validate_data, split_data, split_test_data, scale_features,
             fit_feature_scaler, apply_feature_scaler)
from features.windowing import create_windows

DEFAULT_TRAIN_END = "2021-12-31"
DEFAULT_TEST_START = "2022-01-01"
DEFAULT_OBS_WINDOW_SIZE = 30
DEFAULT_ROLLING_WINDOW = 20


def load_data(train_end, val_start, val_end,
              obs_window_size = DEFAULT_OBS_WINDOW_SIZE,
              rolling_window = DEFAULT_ROLLING_WINDOW):
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

def load_test_data(train_end=DEFAULT_TRAIN_END, test_start=DEFAULT_TEST_START,
    obs_window_size=DEFAULT_OBS_WINDOW_SIZE, rolling_window=DEFAULT_ROLLING_WINDOW,):
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


def load_latest_observation(
    train_end=DEFAULT_TRAIN_END,
    obs_window_size=DEFAULT_OBS_WINDOW_SIZE,
    rolling_window=DEFAULT_ROLLING_WINDOW,
    market_data=None,
):
    """
    Build the latest model observation from current market data.

    Features are scaled with statistics fit only on the historical training
    period used by the saved model. The returned window ends at the most
    recent available feature date (no future rows).

    Parameters
    ----------
    market_data : pd.DataFrame, optional
        Pre-loaded Yahoo Finance-style price data. When omitted, prices are
        downloaded through the latest available date.

    Returns
    -------
    latest_window : np.ndarray
        Shape (obs_window_size, n_features).
    as_of_date : pandas.Timestamp
        Date of the last row in the observation window.
    asset_names : list[str]
        Portfolio asset order, matching training-time returns columns.
    """
    if market_data is None:
        data = load_etf_data(end=None)
    else:
        data = market_data
    data = validate_data(data)
    features, returns = create_features(data, rolling_window)
    train_features = features.loc[:train_end]
    if train_features.empty:
        raise ValueError(
            "No training-period features available to fit the scaler "
            f"(train_end={train_end})."
        )
    scaler = fit_feature_scaler(train_features)
    scaled_features = apply_feature_scaler(features, scaler)
    windows, dates = create_windows(scaled_features, obs_window_size)
    return windows[-1], dates[-1], list(returns.columns)
