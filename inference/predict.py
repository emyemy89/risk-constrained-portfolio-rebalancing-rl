from stable_baselines3 import PPO

from data.pipeline import *
from features.windowing import *
from data.extract.load_data import *


MODEL_PATH = "models/final_model.zip"


def predict_allocation(current_weights):
    data = load_latest_market_data()
    features = prepare_features(data)

    observation = build_observation(
        features=features,
        current_weights=current_weights,
    )

    model = PPO.load(MODEL_PATH)

    action, _ = model.predict(
        observation,
        deterministic=True,
    )

    return action