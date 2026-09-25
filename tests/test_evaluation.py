"""
Tests for src/evaluation.py.

compute_metrics/mean_metrics are pure numpy/pandas and are tested
directly. walk_forward_backtest requires xgboost, which was not
installable in the sandbox this project was built in - that test uses
pytest.importorskip so it runs wherever xgboost is available (including
CI, via requirements.txt) and skips cleanly everywhere else, rather than
silently not existing.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluation import compute_metrics, mean_metrics  # noqa: E402


def test_compute_metrics_perfect_prediction():
    y_true = [10, 20, 30, 40]
    y_pred = [10, 20, 30, 40]
    m = compute_metrics(y_true, y_pred)
    assert m["mae"] == 0
    assert m["rmse"] == 0
    assert m["mape"] == 0


def test_compute_metrics_known_error():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 190.0])
    m = compute_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx(10.0)
    # MAPE = mean(|10/100|, |10/200|) * 100 = mean(10%, 5%) = 7.5%
    assert m["mape"] == pytest.approx(7.5)


def test_compute_metrics_guards_against_zero_true_values():
    """A y_true of exactly 0 would divide by zero in MAPE - must be
    masked out rather than raising or returning inf/nan."""
    y_true = np.array([0.0, 100.0])
    y_pred = np.array([5.0, 110.0])
    m = compute_metrics(y_true, y_pred)
    assert m["mape"] is not None
    assert np.isfinite(m["mape"])


def test_mean_metrics_averages_correctly():
    folds = [
        {"mae": 10.0, "rmse": 12.0, "mape": 5.0},
        {"mae": 20.0, "rmse": 22.0, "mape": 7.0},
    ]
    avg = mean_metrics(folds)
    assert avg["mae"] == pytest.approx(15.0)
    assert avg["rmse"] == pytest.approx(17.0)
    assert avg["mape"] == pytest.approx(6.0)


def test_mean_metrics_handles_empty_list():
    assert mean_metrics([]) == {}


def test_mean_metrics_ignores_none_values():
    folds = [{"mae": 10.0, "mape": None}, {"mae": 20.0, "mape": 8.0}]
    avg = mean_metrics(folds)
    assert avg["mae"] == pytest.approx(15.0)
    assert avg["mape"] == pytest.approx(8.0)  # only the non-None value counted


def test_walk_forward_backtest_runs_with_real_xgboost():
    """Requires xgboost - skipped if not installed. See module docstring."""
    pytest.importorskip("xgboost", reason="xgboost not installed")
    from evaluation import walk_forward_backtest

    n = 24 * 60
    ts = pd.date_range("2026-01-01", periods=n, freq="h")
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "aqi_index": rng.uniform(50, 400, n),
        "some_feature": rng.uniform(0, 1, n),
    })
    y = pd.Series(rng.uniform(50, 400, n))
    seasonal_baseline = pd.Series(rng.uniform(50, 400, n))

    result = walk_forward_backtest(X, y, seasonal_baseline, verbose=False)
    assert result is not None
    assert result["n_folds_run"] > 0
    assert "mae" in result["model"]
    assert len(result["pooled_residuals"]) > 0
