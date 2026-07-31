"""
Generates the next-7-days AQI forecast for every location, using:
  - the latest known history (for lag/rolling features)
  - Open-Meteo's actual weather *forecast* (for future temp/humidity/etc.)
  - the per-(location, horizon) models trained by train.py

Usage:
    python src/forecast.py
"""

import joblib
import pandas as pd
import requests

from config import LOCATIONS, HORIZONS, HISTORY_PATH, FORECAST_PATH, MODELS_DIR
from features import build_inference_row

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_HOURLY_VARS = "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,weather_code"
REQUEST_TIMEOUT = 20


def fetch_future_weather(lat: float, lon: float) -> pd.DataFrame:
    """Hourly weather forecast for the next 7 days."""
    r = requests.get(WEATHER_URL, params={
        "latitude": lat, "longitude": lon,
        "hourly": WEATHER_HOURLY_VARS,
        "forecast_days": 8,
        "timezone": "Asia/Kolkata",
    }, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()["hourly"]
    df = pd.DataFrame(data).rename(columns={
        "time": "timestamp",
        "temperature_2m": "temp_c",
        "relative_humidity_2m": "humidity",
        "pressure_msl": "pressure_mb",
        "wind_speed_10m": "windspeed_kph",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def nearest_future_row(future_df: pd.DataFrame, target_time: pd.Timestamp) -> dict:
    idx = (future_df["timestamp"] - target_time).abs().idxmin()
    row = future_df.loc[idx]
    return {
        "temp_c": row["temp_c"],
        "humidity": row["humidity"],
        "pressure_mb": row["pressure_mb"],
        "windspeed_kph": row["windspeed_kph"],
        "weather_code": row["weather_code"],
    }


def forecast_location(location: str, df_loc: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    future_weather = fetch_future_weather(lat, lon)
    now = df_loc["timestamp"].max()

    rows = []
    for horizon in HORIZONS:
        model_path = MODELS_DIR / f"{location.replace(' ', '_')}_h{horizon}.pkl"
        if not model_path.exists():
            print(f"  [{location} h={horizon}] no trained model yet, skipping.")
            continue

        bundle = joblib.load(model_path)
        model, feature_cols = bundle["model"], bundle["feature_cols"]

        target_time = now + pd.Timedelta(hours=horizon)
        fw = nearest_future_row(future_weather, target_time)

        X_row = build_inference_row(df_loc, fw, feature_cols)
        pred = float(model.predict(X_row)[0])

        # Guardrail: with limited training history, XGBoost can extrapolate
        # to implausible values. Clamp to a generous multiple of the recent
        # observed range so a shaky model can't output a wild spike. This
        # band should be loosened (or removed) once you have several months
        # of history and can trust the model's own extrapolation more.
        recent_max = df_loc["aqi_index"].tail(24 * 30).max()
        recent_min = df_loc["aqi_index"].tail(24 * 30).min()
        upper_bound = max(recent_max * 1.5, recent_max + 50)
        lower_bound = max(0.0, recent_min * 0.5)
        pred = min(max(pred, lower_bound), upper_bound)

        rows.append({
            "location": location,
            "issued_at": now,
            "target_time": target_time,
            "horizon_hours": horizon,
            "predicted_aqi": max(0.0, pred),
            "temp_c": fw["temp_c"],
            "humidity": fw["humidity"],
            "windspeed_kph": fw["windspeed_kph"],
            "weather_code": fw["weather_code"],
        })

    return pd.DataFrame(rows)


def main():
    if not HISTORY_PATH.exists():
        raise SystemExit(f"No history found at {HISTORY_PATH}. Run ingest.py first.")

    history = pd.read_parquet(HISTORY_PATH)
    all_forecasts = []

    for location, (lat, lon) in LOCATIONS.items():
        df_loc = history[history["location"] == location].copy()
        if df_loc.empty:
            continue
        print(f"Forecasting {location}...")
        try:
            f = forecast_location(location, df_loc, lat, lon)
            all_forecasts.append(f)
        except Exception as e:  # noqa: BLE001
            print(f"  WARNING: forecast failed for {location}: {e}")

    if not all_forecasts:
        raise SystemExit("No forecasts produced - have you run train.py yet?")

    result = pd.concat(all_forecasts, ignore_index=True)
    result.to_parquet(FORECAST_PATH, index=False)
    print(f"Saved {len(result)} forecast rows to {FORECAST_PATH}")


if __name__ == "__main__":
    main()
