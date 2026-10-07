"""
fetch_data.py

Fetches hourly air-quality data for a single city from the Open-Meteo
Air Quality API and saves it as a raw CSV file.

This is the first step of the AirWise data pipeline. Later steps will
clean this data, engineer features, and train a model on it.
"""

import logging
from pathlib import Path

import pandas as pd
import requests

# Basic logging setup so we see what's happening instead of just print().
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Open-Meteo's air quality endpoint. No API key required.
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


def fetch_air_quality(latitude: float, longitude: float, city_name: str) -> pd.DataFrame:
    """
    Fetch hourly air quality data for one location.

    Args:
        latitude: Latitude of the city.
        longitude: Longitude of the city.
        city_name: Human-readable name, stored as a column for later
            when we have multiple cities.

    Returns:
        A DataFrame with one row per hour, containing timestamp,
        pollutant levels, and the US AQI.

    Raises:
        requests.RequestException: if the network request fails.
        ValueError: if the API response doesn't contain the data we expect.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi",
        "timezone": "auto",
    }

    logger.info(f"Requesting air quality data for {city_name} ({latitude}, {longitude})")

    try:
        response = requests.get(AIR_QUALITY_URL, params=params, timeout=10)
        response.raise_for_status()  # raises an exception for 4xx/5xx responses
    except requests.RequestException as e:
        logger.error(f"Network request failed: {e}")
        raise

    data = response.json()

    if "hourly" not in data:
        raise ValueError(f"Unexpected API response, no 'hourly' key found: {data}")

    hourly = data["hourly"]

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "pm10": hourly["pm10"],
        "pm2_5": hourly["pm2_5"],
        "carbon_monoxide": hourly["carbon_monoxide"],
        "nitrogen_dioxide": hourly["nitrogen_dioxide"],
        "sulphur_dioxide": hourly["sulphur_dioxide"],
        "ozone": hourly["ozone"],
        "us_aqi": hourly["us_aqi"],
    })
    df["city"] = city_name

    logger.info(f"Fetched {len(df)} hourly rows for {city_name}")
    return df


def save_raw_data(df: pd.DataFrame, output_path: Path) -> None:
    """Save a DataFrame to CSV, creating the parent folder if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved raw data to {output_path}")


if __name__ == "__main__":
    # Delhi's coordinates, as a first test city.
    city_df = fetch_air_quality(latitude=28.6139, longitude=77.2090, city_name="Delhi")

    print("\nPreview of fetched data:")
    print(city_df.head())

    save_raw_data(city_df, Path("ml/data/raw_aqi.csv"))