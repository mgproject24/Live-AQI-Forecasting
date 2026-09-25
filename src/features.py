"""
Feature engineering, refactored from the original notebook into reusable
functions so train.py and forecast.py can never drift out of sync on how
a feature is computed.
"""

import numpy as np
import pandas as pd

from config import NUM_COLS, LAG_HOURS, ROLL_WINDOWS, FUTURE_WEATHER_COLS
from festivals import add_festival_features


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hour/weekday/month + cyclical encodings, same as the original notebook."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df = add_festival_features(df)
    return df


def add_lag_roll_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag + rolling-mean features for AQI, pollutants, and weather."""
    df = df.copy()
    for col in NUM_COLS:
        if col not in df.columns:
            continue
        for lag in LAG_HOURS:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    for window in ROLL_WINDOWS:
        df[f"aqi_roll_{window}"] = df["aqi_index"].rolling(window).mean()
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline for a single location's hourly dataframe.

    Expects columns: timestamp, aqi_index, pm2_5, pm10, co, no2,
    temp_c, humidity, pressure_mb, windspeed_kph, weather_code
    """
    df = add_time_features(df)
    df = add_lag_roll_features(df)
    return df


def build_training_matrix(df: pd.DataFrame, horizon: int):
    """Build (X, y, feature_columns, seasonal_baseline) for one forecast
    horizon (in hours).

    The target is aqi_index at t+horizon. Future weather at t+horizon
    (which will come from Open-Meteo's *forecast* API at inference time)
    is approximated during training by the actual historical weather at
    t+horizon — a standard trick since weather forecasts are reasonably
    accurate a few days out.

    Also computes a weekly-seasonal baseline prediction for each row:
    "AQI at target_time = AQI exactly 7 days before target_time". This is
    only ever a lookup of something already known at time t (never a
    future value), which matters — it's what makes it a fair, causally
    valid comparison point for the model rather than a lucky peek ahead.
    Concretely: target_time - 168h = t + (horizon - 168), so the value
    needed is `horizon` hours in the *past* relative to a full week
    before now... equivalently, a lag of (168 - horizon) hours applied
    to "now" (t). horizon is always <= 168, so this lag is always >= 0.
    """
    feat = build_features(df)

    for col in FUTURE_WEATHER_COLS:
        if col in feat.columns:
            feat[f"{col}_future"] = feat[col].shift(-horizon)

    feat["target"] = feat["aqi_index"].shift(-horizon)

    seasonal_lag = 168 - horizon
    feat["_seasonal_baseline"] = feat["aqi_index"].shift(seasonal_lag)

    feat = feat.dropna().reset_index(drop=True)

    # timestamp/target aren't features; location/lat/lon are identifiers,
    # not predictive signal (lat/lon are constant per location, and
    # location is a non-numeric string XGBoost can't consume directly);
    # _seasonal_baseline is an evaluation comparison point, not a model
    # input (returned separately below, deliberately excluded from X so
    # the model can't just "cheat" by copying the baseline it's compared
    # against).
    drop_cols = {"timestamp", "target", "location", "lat", "lon", "_seasonal_baseline"}
    feature_cols = [c for c in feat.columns if c not in drop_cols]

    X = feat[feature_cols]
    y = feat["target"]
    seasonal_baseline = feat["_seasonal_baseline"]
    return X, y, feature_cols, seasonal_baseline


def build_inference_row(df: pd.DataFrame, future_weather: dict, feature_cols: list) -> pd.DataFrame:
    """Build a single-row feature frame for forecasting, using the latest
    known history plus a dict of future weather values for one horizon.

    future_weather: dict with keys temp_c, humidity, pressure_mb,
    windspeed_kph, weather_code — the forecasted weather at t+horizon.
    """
    feat = build_features(df)
    last_row = feat.iloc[[-1]].copy()

    for col in FUTURE_WEATHER_COLS:
        last_row[f"{col}_future"] = future_weather.get(col, np.nan)

    # Align to the exact column set/order the model was trained on.
    for col in feature_cols:
        if col not in last_row.columns:
            last_row[col] = np.nan
    return last_row[feature_cols]
