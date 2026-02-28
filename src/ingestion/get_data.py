import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from datetime import datetime
import os


def fetch_weather_data():
    # Setup API client with cache to avoid redundant calls
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # API Parameters for the Weather Forecast Project
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 38.801,  # Sintra, Portugal
        "longitude": -9.3783,
        "start_date": "2026-02-06",
        "end_date": "2026-02-20",
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation"],
        "timezone": "auto",
    }

    print(f"Requesting weather data from Open-Meteo...")
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]  # Assuming we only have one response for the given parameters

    # Process hourly data into a dictionary
    hourly = response.Hourly()
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time() + response.UtcOffsetSeconds(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd() + response.UtcOffsetSeconds(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
        "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
        "precipitation": hourly.Variables(2).ValuesAsNumpy()
    }

    df = pd.DataFrame(data=hourly_data)

    # Generate timestamped filename for automatic capturing
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"weather_{timestamp}.csv"
    os.makedirs("data/raw", exist_ok=True)
    output_path = os.path.join("data/raw", filename)

    df.to_csv(output_path, index=False)
    print(f"Data ingested and saved to: {output_path}")


if __name__ == "__main__":
    fetch_weather_data()
