import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
import mlflow
import mlflow.sklearn

# config
params = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42
}
DATA_PATH = "data/processed/final_dataset.csv"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    return df


def train_test_split(df, split_ratio=0.8):
    split_idx = int(len(df) * split_ratio)

    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    return train, test


def get_features(df):
    drop_cols = ["datetime", "demand"]
    features = [col for col in df.columns if col not in drop_cols]
    return features


def train_model(train_df, features):
    X_train = train_df[features]
    y_train = train_df["demand"]

    model = LGBMRegressor(**params)

    model.fit(X_train, y_train)

    return model


def evaluate(model, test_df, features):
    X_test = test_df[features]
    y_test = test_df["demand"]

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return preds, mae, rmse


def save_model(model, path="models/lgbm_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved")


def main():
    df = load_data()

    train_df, test_df = train_test_split(df)

    features = get_features(df)
    print(f"features: {features}")

    # MLflow
    mlflow.set_experiment("energy-demand-forecasting")
    
    with mlflow.start_run(run_name="lgbm_baseline"):
        model = train_model(train_df, features)

        preds, mae, rmse = evaluate(model, test_df, features)

        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_params(params)
        mlflow.log_param("num_features", len(features))
        mlflow.log_param("features", ",".join(features))

        mlflow.sklearn.log_model(model, "model")
        print("MLflow logging complete")

    save_model(model)


if __name__ == "__main__":
    main()
