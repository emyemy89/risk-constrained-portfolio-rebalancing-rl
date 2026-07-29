import joblib
import numpy as np

def compute_ranking_probabilities(windows):
    model = joblib.load("../models/ranking_model.pkl")
    n_samples = windows.shape[0]
    flattened = windows.reshape(n_samples, -1)
    probabilities = model.predict_proba(flattened)
    return probabilities.astype(np.float32)