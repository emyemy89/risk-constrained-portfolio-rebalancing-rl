"""
Data preprocessing utilities for the portfolio pipeline.

This module provides helper functions to align asset price data, validate
datasets.
It is also responsible for splitting data into train, validation, and test periods, and standardize
features for model training.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler


def align_assets(prices):
    """
    Align assets price data to match for all by dropping the nulls
    :param prices: prices for all assets
    :return: prices.dropna()
    """
    return prices.dropna()


def validate_data(data):
    """
    Validate data by checking for null or duplicated values
    :param data: data to be validated
    :return: data
    """
    if data.empty:
        raise ValueError("Input data is empty")
    if data.index.duplicated().sum() > 0:
        raise ValueError("Duplicate index found")
    return data


def split_data(data, train_end, val_start, val_end):
    """
    Split time-series data into chronological training and validation periods.
    data : pd.DataFrame
        Time-indexed feature or return data.
    train_end : str
        Last date included in training.
    val_start : str
        First date included in validation.
    val_end : str
        Last date included in validation.
    Returns
    train_data : pd.DataFrame
    val_data : pd.DataFrame
    """
    train_data = data.loc[:train_end]
    val_data = data.loc[val_start:val_end]
    return train_data, val_data

def get_test_data(data):
    """
    Return the final untouched test period.
    """
    return data.loc['2022-01-01':]


def scale_features(train_features, val_features):
    """
    Scale features based on scaling factors
    :param train_features: The set of concatenated features of assets for training
    :param val_features: The set of concatenated features of assets for validation
    :return: train_scaled, val_scaled, test_scaled
    """
    scaler = StandardScaler()
    scaler.fit(train_features)
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

    return train_scaled, val_scaled
