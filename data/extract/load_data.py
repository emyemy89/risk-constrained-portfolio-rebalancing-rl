import yfinance as yf


def load_etf_data():
    return yf.download(
    ["SPY", "QQQ", "TLT", "GLD", "VNQ"],
            start="2000-01-01",
            end="2026-01-01",
            auto_adjust=True
        )