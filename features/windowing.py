import numpy as np
import pandas as pd


def create_windows(
    data: pd.DataFrame,
    window_size: int,
) -> np.ndarray:
    """
    Convert a feature dataframe into rolling observation windows.
    Parameters
    ----------
    data
        Scaled feature dataframe.
    window_size
        Number of historical timesteps per observation.
    Returns
    -------
    np.ndarray
        Shape:
        (num_samples, window_size, num_features)
    """
    # validation
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if window_size > len(data):
        raise ValueError("window_size larger than dataset")
    values = data.to_numpy()
    windows = []
    for end in range(window_size, len(values) + 1):
        start = end - window_size
        windows.append(values[start:end])
    return np.asarray(windows)