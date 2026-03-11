import hydra
from omegaconf import DictConfig
from typing import Dict, Any
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import logging
import os

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def ingest_data(cfg: DictConfig) -> None:
    """
    Retrieves historical weather data from Open-Meteo API based on Hydra configuration
    and saves it to the specified raw data path.
    """
    logger.info("--- Starting Data Ingestion Pipeline ---")

    # 1. Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # 2. Extract configurations
    location_name = cfg.api.target_location
    coords = cfg.api.locations[location_name]
    dates = cfg.api.date_range
    variables = cfg.api.variables
    output_path = cfg.api.output.raw_data_path

    logger.info(f"Target Location: {location_name} (Lat: {coords.latitude}, Lon: {coords.longitude})")
    logger.info(f"Date Range: {dates.start_date} to {dates.end_date}")

    # 3. Construct the API Payload
    params = {
        "latitude": coords.latitude,
        "longitude": coords.longitude,
        "start_date": dates.start_date,
        "end_date": dates.end_date,
        "hourly": list(variables),
        "timezone": cfg.api.api.timezone
    }

    # 4. Execute the API Request
    logger.info("Fetching data from Open-Meteo Historical API...")
    url = cfg.api.api.endpoint
    responses = openmeteo.weather_api(url, params=params)

    # Process the first location (we only requested one)
    response = responses[0]
    hourly = response.Hourly()

    # 5. Process Data into a Pandas DataFrame
    # Note: Constructing the time index correctly to match the array lengths
    start_time = pd.to_datetime(hourly.Time(), unit="s", utc=True)
    end_time = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True)
    interval = pd.Timedelta(seconds=hourly.Interval())

    # Create the date range - this ensures the index length matches the variables
    date_range = pd.date_range(
        start=start_time,
        end=end_time,
        freq=interval,
        inclusive="left"
    )

    # Initialize the dictionary with the date and the location name
    hourly_data: Dict[str, Any] = {
        "date": date_range,
        "location": [location_name] * len(date_range)  # Broadcast location name to all rows
    }

    # Dynamically map the requested variables to the response arrays
    for idx, var_name in enumerate(variables):
        hourly_data[var_name] = hourly.Variables(idx).ValuesAsNumpy()

    new_df = pd.DataFrame(data=hourly_data)

    # 6. Incremental Loading & Deduplication
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        logger.info(f"Existing dataset found at {output_path}. Appending new data...")
        # Read existing data
        existing_df = pd.read_csv(output_path)

        # Convert date strings back to datetime objects for accurate comparison
        existing_df['date'] = pd.to_datetime(existing_df['date'], utc=True)

        # Concatenate old and new data
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)

        # Drop duplicates based on exact date and location.
        # keep='last' ensures that if we fetch newer data for the same date, it overwrites the old.
        combined_df = combined_df.drop_duplicates(subset=['date', 'location'], keep='last')

        # Sort chronologically
        combined_df = combined_df.sort_values(by=['date']).reset_index(drop=True)

        new_records_count = len(combined_df) - len(existing_df)
        logger.info(f"Added {new_records_count} new unique records.")
        final_df = combined_df
    else:
        logger.info(f"No existing dataset found. Creating new file at {output_path}.")
        final_df = new_df

    # Format the dates cleanly for the CSV
    final_df['date'] = final_df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Save to CSV
    final_df.to_csv(output_path, index=False)

    logger.info(f"Successfully ingested {len(final_df)} records.")
    logger.info(f"Data saved to {output_path}")
    logger.info("--- Data Ingestion Complete ---")


if __name__ == "__main__":
    ingest_data()
