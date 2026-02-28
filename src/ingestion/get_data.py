import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from datetime import datetime
import os
import hydra
from omegaconf import DictConfig, OmegaConf


# The decorator tells Hydra where to find the config files
# config_path is relative to the location of this python script
@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def fetch_weather_data(cfg: DictConfig):
    print(f"Starting ingestion for project: {cfg.project_name}")

    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"

    # Convert the Hydra DictConfig into a standard Python dictionary for the API
    params = OmegaConf.to_container(cfg.api.params, resolve=True)

    print(f"Requesting data for coordinates: {params['latitude']}N, {params['longitude']}E")
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
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

    # Use the path defined in conf/paths/default.yaml
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"weather_{timestamp}.csv"

    # Ensure the directory exists
    os.makedirs(cfg.paths.raw_data_dir, exist_ok=True)
    output_path = os.path.join(cfg.paths.raw_data_dir, filename)

    df.to_csv(output_path, index=False)
    print(f"Data successfully ingested: {output_path}")


if __name__ == "__main__":
    fetch_weather_data()
