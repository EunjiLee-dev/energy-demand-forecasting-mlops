import os
import pandas as pd

def load_energy():
    df = pd.read_csv("data/processed/ch_load_2023_2025.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

def load_weather():
    df = pd.read_csv("data/raw/zurich_weather_2023_2025.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def build_dataset():
    energy = load_energy()
    weather = load_weather()

    df = pd.merge(energy, weather, how="inner", on="datetime")

    df = df.sort_values("datetime")

    # add time features
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month

    # add lag features
    df["lag_1"] = df["demand"].shift(1)
    df["lag_24"] = df["demand"].shift(24)
    df["lag_168"] = df["demand"].shift(168)  # weekly pattern

    # add rolling features
    df["rolling_mean_24"] = df["demand"].rolling(24).mean()
    df["rolling_std_24"] = df["demand"].rolling(24).std()

    df = df.dropna()

    output_path = "data/processed/final_dataset.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    build_dataset()