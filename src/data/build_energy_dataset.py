import pandas as pd
import os

def load_and_clean(file_path):
    df = pd.read_csv(file_path, sep="\t")

    # filter Switzerland
    df = df[df["CountryCode"] == "CH"].copy()

    df["datetime"] = pd.to_datetime(
        df["DateShort"] + " " + df["TimeFrom"],
        dayfirst=True
    )

    df = df[["datetime", "Value"]]

    df = df.rename(columns={"Value":"demand"})

    return df


def build_dataset():
    files = [
        "data/raw/monthly_hourly_load_values_2023.csv",
        "data/raw/monthly_hourly_load_values_2024.csv",
        "data/raw/monthly_hourly_load_values_2025.csv"
    ]

    dfs = []

    for file in files:
        df = load_and_clean(file)
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.sort_values("datetime")
    full_df = full_df.drop_duplicates(subset=["datetime"])

    output_path = "data/processed/ch_load_2023_2025.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    full_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(full_df.head())


if __name__ == "__main__":
    build_dataset()