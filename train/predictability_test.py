import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from data.pipeline import load_training_data

(
        train_windows,
        train_returns,
        val_windows,
        val_returns,
        test_windows,
        test_returns,
    ) = load_training_data()

X = train_windows.reshape(train_windows.shape[0],-1)
X = X[:-1] # align
y = train_returns[1:, 0]

# split chronologically
split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

# train rnd forest
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# evaluate
pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print("RMSE:", rmse)

# compare against naive prediction
baseline_pred = np.zeros_like(y_test)

baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
print("Baseline RMSE:", baseline_rmse)