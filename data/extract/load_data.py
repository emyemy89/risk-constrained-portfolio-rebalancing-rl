"""
Data loading for portfolio pipeline
This module downloads historical price data of assets used in training, validation and testing
"""
import yfinance as yf


def load_etf_data():
    """
    Download historical price data for assets used in the project
    """
    return yf.download(
    ["SPY", "QQQ", "EEM", "GLD", "XLE"], # Cash is loaded in /data/data_engineering/features_def.py
            start="2000-01-01",
            end="2026-01-01",
            auto_adjust=True
        )
