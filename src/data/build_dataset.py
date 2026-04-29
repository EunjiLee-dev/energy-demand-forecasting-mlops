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

    print(df.head())
    print(df.isnull().sum())

    output_path = "data/processed/final_dataset.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    build_dataset()