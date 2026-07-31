"""
Trains a direct multi-horizon AQI forecaster: one XGBoost model per
(location, horizon) pair. Direct horizons avoid the compounding error of
recursively feeding a model its own predictions.

Usage:
    python src/train.py
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from config import LOCATIONS, HORIZONS, HISTORY_PATH, MODELS_DIR
from features import build_training_matrix

MIN_ROWS_REQUIRED = 200  # roughly a week+ of hourly data before training is meaningful


def train_one(df_loc: pd.DataFrame, location: str, horizon: int):
    X, y, feature_cols = build_training_matrix(df_loc, horizon)

    if len(X) < MIN_ROWS_REQUIRED:
        print(f"  [{location} h={horizon}] only {len(X)} usable rows, skipping "
              f"(need >= {MIN_ROWS_REQUIRED}). Keep the daily ingest running "
              f"and retrain later.")
        return None

    split = int(len(X) * 0.85)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    metrics = {"n_train": len(X_train), "n_test": len(X_test)}
    if len(X_test) > 0:
        pred = model.predict(X_test)
        metrics["mae"] = float(mean_absolute_error(y_test, pred))
        metrics["r2"] = float(r2_score(y_test, pred)) if len(X_test) > 1 else None

    bundle = {"model": model, "feature_cols": feature_cols, "metrics": metrics}
    out_path = MODELS_DIR / f"{location.replace(' ', '_')}_h{horizon}.pkl"
    joblib.dump(bundle, out_path)
    print(f"  [{location} h={horizon}] trained on {len(X_train)} rows, "
          f"MAE={metrics.get('mae', 'n/a')}")
    return metrics


def main():
    if not HISTORY_PATH.exists():
        raise SystemExit(
            f"No history found at {HISTORY_PATH}. Run `python src/ingest.py "
            f"--past-days 30` first to backfill data."
        )

    history = pd.read_parquet(HISTORY_PATH)
    all_metrics = {}

    for location in LOCATIONS:
        df_loc = history[history["location"] == location].copy()
        if df_loc.empty:
            print(f"[{location}] no data yet, skipping.")
            continue

        print(f"Training models for {location} ({len(df_loc)} rows)...")
        loc_metrics = {}
        for horizon in HORIZONS:
            m = train_one(df_loc, location, horizon)
            if m:
                loc_metrics[horizon] = m
        all_metrics[location] = loc_metrics

    with open(MODELS_DIR / "training_report.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("Saved training report to models/training_report.json")


if __name__ == "__main__":
    main()
