import requests
import pandas as pd
import os

def fetch_zurich_weather(start_date="2023-01-01", end_date="2025-12-31"):
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": 47.3667, # Zurich
        "longitude": 8.55,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "cloud_cover",
            "shortwave_radiation",
            "apparent_temperature"
        ],
        "timezone": "Europe/Zurich"
    }

    response = requests.get(url, params=params)
    data = response.json()

    # JSON to Dataframe
    df = pd.DataFrame({
        "datetime": data["hourly"]["time"],
        "temperature": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relative_humidity_2m"],
        "precipitation": data["hourly"]["precipitation"],
        "wind_speed": data["hourly"]["wind_speed_10m"],
        "cloud": data["hourly"]["cloud_cover"],
        "shortwave": data["hourly"]["shortwave_radiation"],
        "apparent_temp": data["hourly"]["apparent_temperature"],
    })

    df["datetime"] = pd.to_datetime(df["datetime"])

    return df

def save_weather_data():
    df = fetch_zurich_weather()

    output_path = "data/raw/zurich_weather_2023_2025.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    save_weather_data()