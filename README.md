# risk-constrained-portfolio-rebalancing-rl

This represents a PPO-based portfolio rebalancing agent that learns to allocate across 5 ETFs + cash using risk-adjusted rewards.

## Overview
This project trains a Reinforcement Learning agent (using Proximal Policy Optimization-PPO) to manage a portfolio consisting ofSPY, QQQ, TLT, GLD, VNQ and a synthetic CASH asset. It utilizes a custom Gymnasium environment that incorporates constraints such as transaction costs, turnover limits, and a risk penalty. 
The reward function is shaped around a multi-horizon risk-adjusted return, factoring in asymmetric penalties for losses. Models are trained and evaluated across multiple random seeds for robustness.

## Project Structure
Detailed information for each component can be found in their respective directory `README.md` files.

```text
├── data/          → Data pipeline: download, features, splits (see data/README.md)
├── env/           → Custom Gymnasium environment (see env/README.md)
├── features/      → Rolling window construction
├── train/         → PPO training, evaluation, baselines (see train/README.md)
├── models/        → Saved best + final models (gitignored)
└── logs/          → TensorBoard logs (gitignored)
```

## Key Design Decisions
- **Action Space**: The agent outputs delta-weight changes (limited to ±20% max per step) rather than raw weights, which prevents wild allocation swings and unrealistic turnover.
- **Observation Space**: A 30-day rolling window of 19+ features (log returns, volatility, momentum, SPY correlations, trend regime, drawdown) combined with the current portfolio weights, flattened into a single vector.
- **Reward Function**: Based on a 5-day forward return weighted by the current allocation, combined with an asymmetric loss penalty, transaction cost penalty, and an optional volatility risk penalty (controlled by `risk_lambda`).
- **Data Splits**: Temporal splits to prevent lookahead bias (Train: ≤ 2018, Validation: 2019–2021, Test: 2022+).
- **Feature Scaling**: Features are z-scored independently per split to avoid cross-contamination.
- **CASH Asset**: Treated as a synthetic zero-return asset appended to the ETF universe.

## Quickstart

### 1. Setup
```bash
# Clone the repository
git clone https://github.com/emyemy89/risk-constrained-portfolio-rebalancing-rl.git
cd risk-constrained-portfolio-rebalancing-rl

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Training
Run the training script, which will automatically train the PPO agent across 5 random seeds (~200K steps each) and save the results.
```bash
cd train
python run_training.py
```

### 3. Monitoring
You can monitor training progress via TensorBoard:
```bash
tensorboard --logdir logs/tensorboard
```

## Evaluation & Baselines
The evaluation framework assesses the trained PPO policies against SPY-only and Equal-Weight baselines. Key metrics include:
- **Financial Metrics**: Annualized Return, Annualized Volatility, Sharpe Ratio, and Max Drawdown.
- **Diagnostics**: Weight statistics, portfolio turnover, distance from equal-weight allocation, cash timing analysis, and bull/bear regime allocation.
- **Visualizations**: Plots for portfolio value curves, stacked-area weight charts, and cash allocation over time.

Results, metrics, and plots for each seed are saved under `train/results/seed_N/`.

## License
This project is licensed under the Apache 2.0 License — see the [LICENSE](LICENSE) file for details.