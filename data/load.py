import yfinance as yf


data = yf.download(
    ["SPY", "QQQ", "TLT", "GLD", "VNQ"],
    start="2000-01-01",
    end="2026-01-01",
    auto_adjust=True
)