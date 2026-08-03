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

def split_data(features):
    return( features.loc[:'2018-12-31'],
            features.loc['2019-01-01':'2021-12-31'],
            features.loc['2022-01-01':])

def scale_features(train_features, val_features, test_features):
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
    test_scaled = pd.DataFrame(
        scaler.transform(test_features),
        index=test_features.index,
        columns=test_features.columns,
    )
    return train_scaled, val_scaled, test_scaled
