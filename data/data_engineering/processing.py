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

def scale_features(features):
    scaler = StandardScaler()
    scaler.fit(features)
    return  pd.DataFrame(
        scaler.transform(features),
        index=features.index,
        columns=features.columns,
    )