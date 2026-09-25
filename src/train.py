"""
Trains a direct multi-horizon AQI forecaster: one XGBoost model per
(location, horizon) pair. Direct horizons avoid the compounding error of
recursively feeding a model its own predictions.

See src/evaluation.py for the walk-forward backtest / baseline
methodology. This script adds:

1. Promotion gating - a freshly trained model only overwrites the
   deployed one if it beats the better of the two baselines and isn't
   meaningfully worse than what's already in production.
2. An append-only experiment log for drift monitoring (see monitor.py).
3. Prediction intervals via split conformal prediction: the backtest's
   pooled out-of-sample residuals are used to compute an 80% and 90%
   interval half-width, calibrated per (location, horizon). This is a
   deliberately simple form of conformal prediction - it assumes
   residual magnitude doesn't vary much with the predicted value itself
   (no bucketing by prediction range), which is a real simplification,
   not textbook-complete conformal prediction. It's still meaningfully
   more honest than a point forecast with no uncertainty at all.
4. Optional per-(location, horizon) tuned hyperparameters, loaded from
   models/best_params.json if tune.py has been run. Falls back to
   sensible defaults otherwise.

Usage:
    python src/train.py
"""

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from config import LOCATIONS, HORIZONS, HISTORY_PATH, MODELS_DIR
from features import build_training_matrix
from evaluation import (
    MIN_ROWS_REQUIRED, N_BACKTEST_FOLDS, DEFAULT_XGB_PARAMS,
    walk_forward_backtest,
)

# A retrained model must not be worse than the currently deployed one by
# more than this fraction. 1.10 = allow up to 10% MAE regression before
# blocking promotion (small noise between runs is expected; a genuine
# regression usually shows up as much more than 10%).
PROMOTION_TOLERANCE = 1.10

EXPERIMENT_LOG_PATH = MODELS_DIR / "experiment_log.jsonl"
BEST_PARAMS_PATH = MODELS_DIR / "best_params.json"

# Interval levels to calibrate via conformal prediction.
INTERVAL_LEVELS = [0.80, 0.90]


def _log_experiment(record: dict):
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(EXPERIMENT_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def _load_tuned_params(location: str, horizon: int) -> dict:
    """Returns tuned hyperparameters for this (location, horizon) if
    tune.py has produced them, otherwise the default parameters."""
    if not BEST_PARAMS_PATH.exists():
        return DEFAULT_XGB_PARAMS
    with open(BEST_PARAMS_PATH) as f:
        best_params = json.load(f)
    key = f"{location}|{horizon}"
    entry = best_params.get(key)
    if entry is None:
        return DEFAULT_XGB_PARAMS
    return entry["params"]


def _decide_promotion(new_mae, best_baseline_mae, model_path):
    """Returns (should_promote: bool, reason: str)."""
    if new_mae is None:
        return False, "no backtest MAE available to evaluate the new model"

    if best_baseline_mae is not None and new_mae > best_baseline_mae:
        return False, (
            f"new model (MAE={new_mae:.2f}) does not beat the better of the "
            f"two baselines (MAE={best_baseline_mae:.2f}) - not promoting"
        )

    if model_path.exists():
        old_bundle = joblib.load(model_path)
        old_mae = old_bundle.get("metrics", {}).get("mae")
        if old_mae is not None and new_mae > old_mae * PROMOTION_TOLERANCE:
            return False, (
                f"new model (MAE={new_mae:.2f}) is worse than the deployed "
                f"model (MAE={old_mae:.2f}) beyond the "
                f"{PROMOTION_TOLERANCE:.0%} tolerance - keeping deployed model"
            )

    return True, "beat both baselines and passed the regression check"


def _conformal_intervals(pooled_residuals: list) -> dict:
    """Half-widths for each interval level in INTERVAL_LEVELS, from the
    empirical quantiles of |residual| pooled across all backtest folds.
    See the module docstring for the simplifying assumption this makes."""
    if not pooled_residuals:
        return {}
    abs_resid = np.abs(np.asarray(pooled_residuals, dtype=float))
    return {
        f"interval_{int(level*100)}_halfwidth": float(np.quantile(abs_resid, level))
        for level in INTERVAL_LEVELS
    }


def train_one(df_loc: pd.DataFrame, location: str, horizon: int):
    X, y, feature_cols, seasonal_baseline = build_training_matrix(df_loc, horizon)

    if len(X) < MIN_ROWS_REQUIRED:
        print(f"  [{location} h={horizon}] only {len(X)} usable rows, skipping "
              f"(need >= {MIN_ROWS_REQUIRED}). Keep the daily ingest running "
              f"and retrain later.")
        _log_experiment({
            "location": location, "horizon": horizon, "status": "skipped_insufficient_data",
            "n_rows": len(X),
        })
        return None

    xgb_params = _load_tuned_params(location, horizon)
    backtest = walk_forward_backtest(X, y, seasonal_baseline, xgb_params=xgb_params)
    if backtest is None:
        print(f"  [{location} h={horizon}] not enough data for a {N_BACKTEST_FOLDS}-fold "
              f"backtest yet ({len(X)} rows) - skipping this run.")
        _log_experiment({
            "location": location, "horizon": horizon, "status": "skipped_insufficient_data_for_backtest",
            "n_rows": len(X),
        })
        return None

    model_mae = backtest["model"]["mae"]
    naive_mae = backtest["naive_baseline"]["mae"]
    seasonal_mae = backtest["seasonal_baseline"]["mae"]
    best_baseline_mae = min(v for v in (naive_mae, seasonal_mae) if v is not None)

    out_path = MODELS_DIR / f"{location.replace(' ', '_')}_h{horizon}.pkl"
    should_promote, reason = _decide_promotion(model_mae, best_baseline_mae, out_path)

    intervals = _conformal_intervals(backtest["pooled_residuals"])

    metrics = {
        "mae": model_mae, "rmse": backtest["model"]["rmse"], "mape": backtest["model"]["mape"],
        "naive_baseline_mae": naive_mae, "seasonal_baseline_mae": seasonal_mae,
        "beats_naive_baseline": model_mae < naive_mae if naive_mae is not None else None,
        "beats_seasonal_baseline": model_mae < seasonal_mae if seasonal_mae is not None else None,
        "n_backtest_folds": backtest["n_folds_run"],
        "n_rows": len(X),
        "promoted": should_promote,
        "promotion_reason": reason,
        "xgb_params": xgb_params,
        **intervals,
    }

    if should_promote:
        # Final production model: fit on ALL available data, not the
        # held-out backtest split, since a deployed model should use
        # every row it has. The reported metrics above still come from
        # the honest, held-out backtest - not from this fit.
        final_model = XGBRegressor(**xgb_params)
        final_model.fit(X, y)
        bundle = {"model": final_model, "feature_cols": feature_cols, "metrics": metrics}
        joblib.dump(bundle, out_path)
        interval_90 = intervals.get("interval_90_halfwidth")
        interval_str = f", 90% interval=+/-{interval_90:.1f}" if interval_90 is not None else ""
        print(f"  [{location} h={horizon}] PROMOTED - backtest MAE={model_mae:.2f} "
              f"(naive={naive_mae:.2f}, seasonal={seasonal_mae:.2f}), "
              f"RMSE={metrics['rmse']:.2f}, MAPE={metrics['mape']:.1f}%{interval_str}")
    else:
        print(f"  [{location} h={horizon}] NOT promoted - {reason}")

    _log_experiment({"location": location, "horizon": horizon, "status": "trained", **metrics})
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
    print(f"Appended run details to {EXPERIMENT_LOG_PATH}")


if __name__ == "__main__":
    main()
