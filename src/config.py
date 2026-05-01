MODEL_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42
}

FEATURES = [
    # time
    "hour",
    "dayofweek",
    # lag
    "lag_24",
    "lag_168",
    # rolling
    "rolling_mean_24",
    # weather
    "temperature",
    "precipitation",
    "wind_speed"   
]

DATA_PATH = "data/processed/final_dataset.csv"
