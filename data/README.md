# data 📈

Prepares market data for training and evaluation. Everything downstream (env, training) expects precomputed feature windows and aligned asset returns from `load_training_data()` in `pipeline.py`.

## Files

- `load_data.py` — download daily ETF prices (SPY, QQQ, TLT, GLD, VNQ) via yfinance
- `processing.py` — functions meant to validate, engineer features, split by date, scale
- `pipeline.py` — run the full flow and return train/val/test arrays

## Pipeline

The main steps look like:

1. **Download** — fetch adjusted close prices from 2000-01-01 to 2026-01-01
2. **Validate** — reject empty data or duplicate dates
3. **Align** — drop rows with missing prices so all assets share the same calendar
4. **Features** — from aligned closes:
   - log returns: `log(P_t / P_{t-1})`
   - 20-day rolling volatility (recent risk, sth like a month)
   - 20-day rolling momentum (recent direction)
5. **Split** — fixed date ranges to avoid lookahead:
   - train: through 2018
   - validation: 2019–2021
   - test: 2022 onward
6. **Scale** — z-score each split independently so features have comparable magnitude for the policy
7. **Window** — build 30-day rolling observation tensors `(samples, 30, features)` via `features/windowing.py`; the agent sees recent history, not just today
8. **Align returns** — shift returns by `window_size - 1` so each observation lines up with the next-day asset returns used by the environment

## Output

`load_training_data()` returns:

```
train_windows, train_returns,
val_windows,   val_returns,
test_windows,  test_returns
```

Later used by `train/run_training.py` to build `PortfolioEnv` instances
