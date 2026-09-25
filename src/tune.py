"""
Periodic hyperparameter search for each (location, horizon) model, using
the same walk-forward backtest as train.py so tuning is judged by the
same honest, held-out metric the model is actually deployed on.

This is deliberately a small grid search, not Optuna/Bayesian
optimization - it needs zero new dependencies (Optuna wasn't
installable offline in the environment this project was built in, and
a grid search this small doesn't need it anyway), and it's easy to
read and extend by hand.

Why this is a SEPARATE script from train.py, run less often: grid
search multiplies training cost by len(PARAM_GRID) for every
(location, horizon) pair. Running that daily alongside the regular
retrain would slow down (and eventually rate-limit) the daily pipeline
for very little benefit - hyperparameters that were good last week are
still almost certainly fine today. Run this weekly instead (see
.github/workflows/tune.yml).

Usage:
    python src/tune.py
"""

import json

import pandas as pd

from config import LOCATIONS, HORIZONS, HISTORY_PATH, MODELS_DIR
from features import build_training_matrix
from evaluation import walk_forward_backtest, DEFAULT_XGB_PARAMS

BEST_PARAMS_PATH = MODELS_DIR / "best_params.json"

# Deliberately small - each entry costs a full 5-fold backtest per
# (location, horizon). 4 candidates x 48 locations x 7 horizons is
# already ~1,300 backtests; keep this grid small unless you're prepared
# to also raise the workflow's timeout.
PARAM_GRID = [
    dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=-1),
    DEFAULT_XGB_PARAMS,
    dict(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, n_jobs=-1),
    dict(n_estimators=800, max_depth=4, learning_rate=0.02, subsample=0.9, colsample_bytree=0.9, n_jobs=-1),
]


def tune_one(df_loc: pd.DataFrame, location: str, horizon: int):
    X, y, feature_cols, seasonal_baseline = build_training_matrix(df_loc, horizon)

    best = None
    for params in PARAM_GRID:
        backtest = walk_forward_backtest(X, y, seasonal_baseline, xgb_params=params, verbose=False)
        if backtest is None:
            continue
        mae = backtest["model"]["mae"]
        if best is None or mae < best["mae"]:
            best = {"mae": mae, "params": params}

    return best


def main():
    if not HISTORY_PATH.exists():
        raise SystemExit(f"No history found at {HISTORY_PATH}. Run ingest.py first.")

    history = pd.read_parquet(HISTORY_PATH)
    best_params = {}
    if BEST_PARAMS_PATH.exists():
        with open(BEST_PARAMS_PATH) as f:
            best_params = json.load(f)

    for location in LOCATIONS:
        df_loc = history[history["location"] == location].copy()
        if df_loc.empty:
            continue

        print(f"Tuning {location}...")
        for horizon in HORIZONS:
            result = tune_one(df_loc, location, horizon)
            key = f"{location}|{horizon}"
            if result is None:
                print(f"  h={horizon}: not enough data to tune yet, leaving as-is")
                continue
            best_params[key] = result
            print(f"  h={horizon}: best MAE={result['mae']:.2f} with {result['params']}")

    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Saved tuned parameters to {BEST_PARAMS_PATH}")
    print("Run train.py next to deploy models using these tuned parameters.")


if __name__ == "__main__":
    main()
