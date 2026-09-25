"""
Shared evaluation logic used by both train.py (daily retrain) and
tune.py (periodic hyperparameter search). Kept separate so the two
scripts can't drift into two different definitions of "backtest".
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

MIN_ROWS_REQUIRED = 200        # roughly a week+ of hourly data before training is meaningful
N_BACKTEST_FOLDS = 5
MIN_TRAIN_FRACTION = 0.5        # first fold trains on at least this fraction of the data
MIN_FOLD_TEST_ROWS = 5          # skip a fold if its test window would be tiny

DEFAULT_XGB_PARAMS = dict(
    n_estimators=500, learning_rate=0.03, max_depth=4,
    subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
)


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    mask = np.abs(y_true) > 1e-6
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else None

    return {"mae": mae, "rmse": rmse, "mape": mape}


def mean_metrics(list_of_metric_dicts: list) -> dict:
    if not list_of_metric_dicts:
        return {}
    keys = list_of_metric_dicts[0].keys()
    out = {}
    for k in keys:
        vals = [m[k] for m in list_of_metric_dicts if m.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    return out


def walk_forward_backtest(X, y, seasonal_baseline, xgb_params=None,
                           n_folds=N_BACKTEST_FOLDS, verbose=True):
    """Time-ordered expanding-window backtest. Returns a dict with averaged
    model / naive-baseline / seasonal-baseline metrics across folds, the
    pooled out-of-sample residuals (for conformal prediction intervals),
    and per-fold detail for transparency - or None if there isn't enough
    data to backtest meaningfully.

    The first fold's training window is guaranteed to be at least
    MIN_ROWS_REQUIRED rows - without this, a dataset only slightly above
    MIN_ROWS_REQUIRED can silently starve every fold except the last one
    (e.g. 241 total rows produced only 1 usable fold instead of 5 before
    this fix, with no warning). Better to run fewer, well-formed folds
    and say so explicitly than to silently degrade backtest rigor.
    """
    xgb_params = xgb_params or DEFAULT_XGB_PARAMS
    from xgboost import XGBRegressor  # lazy import: keeps compute_metrics/mean_metrics
                                       # usable and testable without xgboost installed
    n = len(X)
    initial_train_end = max(int(n * MIN_TRAIN_FRACTION), MIN_ROWS_REQUIRED)
    remaining = n - initial_train_end
    if remaining < MIN_FOLD_TEST_ROWS:
        return None
    fold_size = max(remaining // n_folds, MIN_FOLD_TEST_ROWS)

    model_folds, naive_folds, seasonal_folds = [], [], []
    fold_details = []
    pooled_residuals = []

    test_start = initial_train_end
    while test_start < n:
        train_end = test_start
        test_end = min(test_start + fold_size, n)

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_test, y_test = X.iloc[test_start:test_end], y.iloc[test_start:test_end]
        seasonal_test = seasonal_baseline.iloc[test_start:test_end]

        if len(X_train) >= MIN_ROWS_REQUIRED and len(X_test) >= MIN_FOLD_TEST_ROWS:
            model = XGBRegressor(**xgb_params)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            model_metrics = compute_metrics(y_test, pred)
            naive_metrics = compute_metrics(y_test, X_test["aqi_index"])
            seasonal_metrics = compute_metrics(y_test, seasonal_test)

            model_folds.append(model_metrics)
            naive_folds.append(naive_metrics)
            seasonal_folds.append(seasonal_metrics)
            pooled_residuals.extend((np.asarray(y_test, dtype=float) - np.asarray(pred, dtype=float)).tolist())
            fold_details.append({
                "fold": len(fold_details), "n_train": len(X_train), "n_test": len(X_test),
                "model": model_metrics, "naive_baseline": naive_metrics,
                "seasonal_baseline": seasonal_metrics,
            })

        test_start = test_end

    if not model_folds:
        return None

    if verbose and len(model_folds) < n_folds:
        print(f"    NOTE: only {len(model_folds)}/{n_folds} backtest folds had enough "
              f"data to run (n={n} rows). Results are still honest, just based on "
              f"fewer folds than usual - this will improve as more history accumulates.")

    return {
        "n_folds_run": len(model_folds),
        "model": mean_metrics(model_folds),
        "naive_baseline": mean_metrics(naive_folds),
        "seasonal_baseline": mean_metrics(seasonal_folds),
        "fold_details": fold_details,
        "pooled_residuals": pooled_residuals,
    }
