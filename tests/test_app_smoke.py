"""
Smoke test for app.py using Streamlit's built-in headless test harness
(streamlit.testing.v1.AppTest). This runs the *entire* dashboard script
and fails if anything raises an uncaught exception during a normal run.

Why this test exists: a Plotly/dependency version mismatch once broke the
map section in production (an AttributeError inside px.scatter_mapbox)
and it wasn't caught until it was live on Streamlit Cloud. This test
exists specifically so that class of bug fails CI instead of reaching
production.

NOTE: this test requires the `streamlit` package, which was not
installable in the sandbox this project was built in (no internet
access there). It has been written carefully against the documented
AppTest API but has not been executed. Run it locally or let CI run it
before trusting it fully - if the AppTest API has changed since,
str(at.exception) will tell you exactly what to fix.

Run with:
    pytest tests/test_app_smoke.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import LOCATIONS, LAST_30_PATH, FORECAST_PATH  # noqa: E402


def _write_fixture_data():
    """Write small, valid history + forecast files so the dashboard has
    something to render. Overwrites whatever is in data/ - fine for a
    CI checkout, but don't run this against your real local data/ dir
    unless you don't mind it being replaced with synthetic values."""
    rng = np.random.default_rng(0)
    location = next(iter(LOCATIONS))
    lat, lon = LOCATIONS[location]

    ts = pd.date_range("2026-01-01", periods=48, freq="h")
    last_30 = pd.DataFrame({
        "timestamp": ts,
        "location": location,
        "lat": lat,
        "lon": lon,
        "aqi_index": rng.uniform(50, 300, 48),
        "pm2_5": rng.uniform(10, 200, 48),
        "pm10": rng.uniform(20, 300, 48),
        "co": rng.uniform(0.1, 2, 48),
        "no2": rng.uniform(5, 80, 48),
        "temp_c": rng.uniform(15, 35, 48),
        "humidity": rng.uniform(20, 90, 48),
        "pressure_mb": rng.uniform(1000, 1020, 48),
        "windspeed_kph": rng.uniform(0, 20, 48),
        "weather_code": rng.integers(0, 4, 48),
    })

    horizons = [24, 48, 72, 96, 120, 144, 168]
    forecast = pd.DataFrame({
        "location": [location] * len(horizons),
        "issued_at": [ts[-1]] * len(horizons),
        "target_time": [ts[-1] + pd.Timedelta(hours=h) for h in horizons],
        "horizon_hours": horizons,
        "predicted_aqi": rng.uniform(50, 300, len(horizons)),
        "temp_c": rng.uniform(15, 35, len(horizons)),
        "humidity": rng.uniform(20, 90, len(horizons)),
        "windspeed_kph": rng.uniform(0, 20, len(horizons)),
        "weather_code": rng.integers(0, 4, len(horizons)),
    })

    LAST_30_PATH.parent.mkdir(parents=True, exist_ok=True)
    last_30.to_parquet(LAST_30_PATH, index=False)
    forecast.to_parquet(FORECAST_PATH, index=False)


def test_dashboard_runs_without_raising():
    streamlit_testing = pytest.importorskip(
        "streamlit.testing.v1",
        reason="streamlit not installed - install streamlit to run this smoke test",
    )
    AppTest = streamlit_testing.AppTest

    _write_fixture_data()

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()

    assert not at.exception, (
        f"Dashboard raised on a normal run: {[str(e) for e in at.exception]}"
    )
