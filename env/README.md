📦 Core responsibilities

This part performs only do 5 things:

1. Serve observations
return X_t = (window_size, num_features)
already precomputed → just index into windows

2. Accept action (portfolio weights)
vector of size N_assets = 5
must be:
- ≥ 0
- sum to 1

3. Simulate portfolio return
apply weights to asset returns at time t+1

4. Track wealth evolution
wealth_{t+1} = wealth_t * (1 + portfolio_return)

5. Compute reward
risk-adjusted return (we’ll start simple)