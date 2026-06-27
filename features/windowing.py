import numpy as np
import pandas as pd


def create_windows(
    data: pd.DataFrame,
    window_size: int,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
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
    if data.empty or len(data) == 0:
        raise ValueError("Input data is empty")
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if window_size > len(data):
        raise ValueError("window_size larger than dataset")

    values = data.to_numpy(dtype=np.float32)
    dates = data.index[window_size - 1:]
    windows = []
    for end in range(window_size, len(values) + 1):
        start = end - window_size
        windows.append(values[start:end])
    return np.asarray(windows, dtype=np.float32), dates


def window_generator(
    data: pd.DataFrame,
    window_size: int,
):
    """
    Memory-efficient rolling window generator.
    Yields one window at a time instead of storing all in memory.
    """
    values = data.to_numpy(dtype=np.float32)
    dates = data.index[window_size - 1:]
    for i in range(window_size - 1, len(values)):
        yield values[i - window_size + 1 : i + 1], dates[i]