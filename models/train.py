import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os


def load_data(path="data/processed/final_dataset.csv"):
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

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31
    )

    model.fit(X_train, y_train)

    return model


def evaluate(model, test_df, features):
    X_test = test_df[features]
    y_test = test_df["demand"]

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    return preds


def save_model(model, path="models/lgbm_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved")


def main():
    df = load_data()

    train_df, test_df = train_test_split(df)

    features = get_features(df)
    print(f"features: {features}")

    model = train_model(train_df, features)

    evaluate(model, test_df, features)

    save_model(model)


if __name__ == "__main__":
    main()
