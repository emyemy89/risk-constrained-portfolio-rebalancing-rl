# Pipeline orchestrator
# %%
from data.extract.load_data import load_etf_data
from data.data_engineering.processing import *
from data.data_engineering.features_def import create_features
from features.windowing import create_windows
from features.ranking_features import compute_ranking_probabilities

# %%
def load_training_data():
    OBSERVATION_WINDOW_SIZE = 30
    ROLLING_WINDOW = 20
    data = load_etf_data()

    # Validate
    data = validate_data(data)

    # %%
    #  Align assets -> Convert raw to log returns -> Calculate volatility and momentum -> Concatenate
    features, returns = create_features(data,ROLLING_WINDOW)
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
    train_windows, train_dates = create_windows(train_scaled, OBSERVATION_WINDOW_SIZE)
    val_windows, val_dates = create_windows(val_scaled, OBSERVATION_WINDOW_SIZE)
    test_windows, test_dates = create_windows(test_scaled, OBSERVATION_WINDOW_SIZE)

    # Compute ranking probabilities
    train_ranking_probs = compute_ranking_probabilities(train_windows)
    val_ranking_probs = compute_ranking_probabilities(val_windows)
    test_ranking_probs = compute_ranking_probabilities(test_windows)

    # return must match the observation, so we shift
    offset = OBSERVATION_WINDOW_SIZE - 1

    train_returns = train_returns.iloc[offset:].to_numpy()
    val_returns = val_returns.iloc[offset:].to_numpy()
    test_returns = test_returns.iloc[offset:].to_numpy()
    # %%
    return (
        train_windows,
        train_returns,
        train_ranking_probs,
        val_windows,
        val_returns,
        val_ranking_probs,
        test_windows,
        test_returns,
        test_ranking_probs,
    )