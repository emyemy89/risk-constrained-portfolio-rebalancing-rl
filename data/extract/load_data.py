"""
Data loading for portfolio pipeline
This module downloads historical price data of assets used in training, validation and testing
"""
import yfinance as yf

# Cash is added later in data/data_engineering/features_def.py
ETF_TICKERS = ("SPY", "QQQ", "EEM", "GLD", "XLE")
DEFAULT_START = "2000-01-01"
DEFAULT_END = "2026-01-01"


def load_etf_data(start=DEFAULT_START, end=DEFAULT_END):
    """
    Download historical price data for assets used in the project.

    Training and evaluation keep the default ``end`` cutoff. Pass ``end=None``
    to download through the latest available market date for inference.
    """
    download_kwargs = {
        "tickers": list(ETF_TICKERS),
        "start": start,
        "auto_adjust": True,
    }
    if end is not None:
        download_kwargs["end"] = end
    return yf.download(**download_kwargs)
