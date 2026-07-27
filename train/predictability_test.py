import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from data.pipeline import load_training_data

(
        train_windows,
        train_returns,
        val_windows,
        val_returns,
        test_windows,
        test_returns,
    ) = load_training_data()

horizon = 20 # prediction over 20 days
future_train_returns = np.array([
    np.sum(train_returns[i:i+horizon, 0])
    for i in range(len(train_returns)-horizon)
])

y_train = (future_train_returns > 0).astype(int)

future_val_returns = np.array([
    np.sum(val_returns[i:i+horizon, 0])
    for i in range(len(val_returns)-horizon)
])

y_val = (future_val_returns > 0).astype(int)

X_train = train_windows.reshape(train_windows.shape[0],-1)
X_val = val_windows.reshape(val_windows.shape[0],-1)
X_val = X_val[:len(y_val)]

X_train = X_train[:len(y_train)]
X_val = X_val[:len(y_val)]





# train rnd forest
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1,)
model.fit(X_train, y_train)

# evaluate
pred = model.predict(X_val)
accuracy = accuracy_score(y_val, pred)
print("Accuracy:", accuracy)

# compare against naive prediction
majority_class = np.bincount(y_val).argmax()
baseline_pred = np.full(len(y_val), majority_class)
baseline_accuracy = accuracy_score(y_val, baseline_pred)

print("Baseline accuracy:", baseline_accuracy)
