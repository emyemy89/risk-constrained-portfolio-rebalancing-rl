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

X_train = train_windows.reshape(train_windows.shape[0],-1)
X_train = X_train[:-1] # align
y_train = train_returns[1:, 0]


X_val = val_windows.reshape(val_windows.shape[0],-1)
y_val = val_returns[1:, 0]

# train rnd forest
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# evaluate
pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, pred))
print("RMSE:", rmse)

# compare against naive prediction
baseline_pred = np.zeros_like(y_val)

baseline_rmse = np.sqrt(mean_squared_error(y_val, baseline_pred))
print("Baseline RMSE:", baseline_rmse)
