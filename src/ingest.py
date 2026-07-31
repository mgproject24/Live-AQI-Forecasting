"""
Pulls live weather + air quality data for every configured location from
Open-Meteo (free, no API key needed) and appends it to the local archive.

Usage:
    python src/ingest.py                  # daily incremental run (past 2 days, overlap-safe)
    python src/ingest.py --past-days 30   # one-time backfill to seed history

Open-Meteo's `past_days` parameter lets us pull recent history and the
live/forecast value in a single call, which keeps this script simple and
free of any API key management.
"""

import argparse
import sys
import time

import pandas as pd
import requests

from config import LOCATIONS, HISTORY_PATH, LAST_30_PATH, DASHBOARD_WINDOW_DAYS

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

WEATHER_HOURLY_VARS = "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,weather_code"
AIR_QUALITY_HOURLY_VARS = "us_aqi,pm2_5,pm10,carbon_monoxide,nitrogen_dioxide"

REQUEST_TIMEOUT = 20


def _get_json(url, params, retries=3):
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001 - want to retry on anything transient
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts: {last_err}")


def fetch_location_history(name: str, lat: float, lon: float, past_days: int) -> pd.DataFrame:
    """Fetch hourly weather + air quality for one location and merge them."""
    weather = _get_json(WEATHER_URL, {
        "latitude": lat, "longitude": lon,
        "hourly": WEATHER_HOURLY_VARS,
        "past_days": past_days, "forecast_days": 1,
        "timezone": "Asia/Kolkata",
    })
    air = _get_json(AIR_QUALITY_URL, {
        "latitude": lat, "longitude": lon,
        "hourly": AIR_QUALITY_HOURLY_VARS,
        "past_days": past_days, "forecast_days": 1,
        "timezone": "Asia/Kolkata",
    })

    w = pd.DataFrame(weather["hourly"])
    a = pd.DataFrame(air["hourly"])

    w = w.rename(columns={
        "time": "timestamp",
        "temperature_2m": "temp_c",
        "relative_humidity_2m": "humidity",
        "pressure_msl": "pressure_mb",
        "wind_speed_10m": "windspeed_kph",
        "weather_code": "weather_code",
    })
    a = a.rename(columns={
        "time": "timestamp",
        "us_aqi": "aqi_index",
        "pm2_5": "pm2_5",
        "pm10": "pm10",
        "carbon_monoxide": "co",
        "nitrogen_dioxide": "no2",
    })

    df = w.merge(a, on="timestamp", how="inner")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["location"] = name
    df["lat"] = lat
    df["lon"] = lon

    # Drop rows where the AQI/weather hasn't landed yet (Open-Meteo sometimes
    # returns nulls for the most recent 1-2 hours before sensors catch up).
    df = df.dropna(subset=["aqi_index", "temp_c"]).reset_index(drop=True)
    return df


def run(past_days: int) -> pd.DataFrame:
    frames = []
    for name, (lat, lon) in LOCATIONS.items():
        print(f"Fetching {name} (past_days={past_days})...")
        try:
            frames.append(fetch_location_history(name, lat, lon, past_days))
        except Exception as e:  # noqa: BLE001
            print(f"  WARNING: failed to fetch {name}: {e}", file=sys.stderr)

    if not frames:
        raise RuntimeError("No data fetched for any location - check network/API status.")

    new_data = pd.concat(frames, ignore_index=True)

    if HISTORY_PATH.exists():
        history = pd.read_parquet(HISTORY_PATH)
        combined = pd.concat([history, new_data], ignore_index=True)
    else:
        combined = new_data

    combined = combined.drop_duplicates(subset=["location", "timestamp"], keep="last")
    combined = combined.sort_values(["location", "timestamp"]).reset_index(drop=True)
    combined.to_parquet(HISTORY_PATH, index=False)

    cutoff = combined["timestamp"].max() - pd.Timedelta(days=DASHBOARD_WINDOW_DAYS)
    last_30 = combined[combined["timestamp"] >= cutoff].reset_index(drop=True)
    last_30.to_parquet(LAST_30_PATH, index=False)

    print(f"History now has {len(combined)} rows across {combined['location'].nunique()} locations.")
    print(f"Last-{DASHBOARD_WINDOW_DAYS}-days slice has {len(last_30)} rows.")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--past-days", type=int, default=2,
        help="How many past days to (re-)fetch. Use a large value (e.g. 30-92) "
             "once to backfill history; the default of 2 is enough for daily "
             "incremental runs since duplicates are dropped automatically.",
    )
    args = parser.parse_args()
    run(args.past_days)
