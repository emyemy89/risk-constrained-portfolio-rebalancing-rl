import yfinance as yf

# Raw Prices
data = yf.download(
    ["SPY", "QQQ", "TLT", "GLD", "VNQ"],
    start="2000-01-01",
    end="2026-01-01",
    auto_adjust=True
)

# 2. Validate

# missing values
data.isna().sum()

# 3. Align assets

# 4. Convert raw to log returns

# 5. Create rolling observation window

# 6. Generate derived features

# 7. Normalize

# 8. Create Splits
