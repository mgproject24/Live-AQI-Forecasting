"""
Shared configuration for the Delhi AQI live forecasting pipeline.
Every script (ingest, features, train, forecast, app) imports from here so
that column names, locations, and horizons never drift out of sync.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

HISTORY_PATH = DATA_DIR / "history.parquet"       # full archive, never trimmed
LAST_30_PATH = DATA_DIR / "last_30_days.parquet"  # what the dashboard reads
FORECAST_PATH = DATA_DIR / "forecast.parquet"      # latest 7-day predictions

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Monitored locations (approximate coordinates for the 6 Delhi stations used
# in the original dataset). Edit/add freely.
# ---------------------------------------------------------------------------
LOCATIONS = {
    "IGI Airport":       (28.5562, 77.1000),
    "Anand Vihar":       (28.6469, 77.3151),
    "Connaught Place":   (28.6315, 77.2167),
    "Dwarka":            (28.5921, 77.0460),
    "Okhla Phase III":   (28.5355, 77.2910),
    "Rohini":            (28.7041, 77.1025),
}

# ---------------------------------------------------------------------------
# Feature engineering config
# ---------------------------------------------------------------------------
# Numeric columns we lag/roll for AQI + pollutant + weather history.
NUM_COLS = [
    "aqi_index", "pm2_5", "pm10", "co", "no2",
    "temp_c", "humidity", "pressure_mb", "windspeed_kph",
]

LAG_HOURS = [1, 24]
ROLL_WINDOWS = [24, 168]

# Weather variables used as "future" inputs (their forecast values at t+h
# stand in for what Open-Meteo's forecast API will give us at inference time).
FUTURE_WEATHER_COLS = ["temp_c", "humidity", "pressure_mb", "windspeed_kph", "weather_code"]

# How far ahead we forecast, in hours. 24h steps => a 7-day-ahead dashboard.
HORIZONS = [24, 48, 72, 96, 120, 144, 168]

# Rolling window (in days) kept for the dashboard's "recent history" view.
DASHBOARD_WINDOW_DAYS = 30

# ---------------------------------------------------------------------------
# AQI category bands (US AQI scale, since Open-Meteo's us_aqi field is used
# as the live `aqi_index` target). Swap this if you switch to CPCB's index.
# ---------------------------------------------------------------------------
AQI_BANDS = [
    (0, 50, "Good", "green"),
    (51, 100, "Moderate", "amber"),
    (101, 150, "Unhealthy (sensitive)", "coral"),
    (151, 200, "Unhealthy", "red"),
    (201, 300, "Very unhealthy", "pink"),
    (301, 500, "Hazardous", "gray"),
]


def aqi_category(value: float):
    """Return (label, color_key) for a given AQI value."""
    if value is None or value != value:  # NaN check without importing numpy
        return "Unknown", "gray"
    for low, high, label, color in AQI_BANDS:
        if low <= value <= high:
            return label, color
    return "Hazardous", "gray"


# ---------------------------------------------------------------------------
# Weather icons — mapped from Open-Meteo's WMO weather_code, using plain
# emoji glyphs (renders everywhere, no external assets/licensing to worry
# about, no network call needed).
# ---------------------------------------------------------------------------
WEATHER_CODE_MAP = {
    0: ("\u2600\ufe0f", "Clear sky"),
    1: ("\U0001F324\ufe0f", "Mainly clear"),
    2: ("\u26c5", "Partly cloudy"),
    3: ("\u2601\ufe0f", "Overcast"),
    45: ("\U0001F32B\ufe0f", "Fog"),
    48: ("\U0001F32B\ufe0f", "Depositing rime fog"),
    51: ("\U0001F326\ufe0f", "Light drizzle"),
    53: ("\U0001F326\ufe0f", "Moderate drizzle"),
    55: ("\U0001F327\ufe0f", "Dense drizzle"),
    56: ("\U0001F327\ufe0f", "Light freezing drizzle"),
    57: ("\U0001F327\ufe0f", "Dense freezing drizzle"),
    61: ("\U0001F327\ufe0f", "Slight rain"),
    63: ("\U0001F327\ufe0f", "Moderate rain"),
    65: ("\U0001F327\ufe0f", "Heavy rain"),
    66: ("\U0001F327\ufe0f", "Light freezing rain"),
    67: ("\U0001F327\ufe0f", "Heavy freezing rain"),
    71: ("\U0001F328\ufe0f", "Slight snow"),
    73: ("\U0001F328\ufe0f", "Moderate snow"),
    75: ("\u2744\ufe0f", "Heavy snow"),
    77: ("\u2744\ufe0f", "Snow grains"),
    80: ("\U0001F326\ufe0f", "Slight showers"),
    81: ("\U0001F327\ufe0f", "Moderate showers"),
    82: ("\u26c8\ufe0f", "Violent showers"),
    85: ("\U0001F328\ufe0f", "Slight snow showers"),
    86: ("\u2744\ufe0f", "Heavy snow showers"),
    95: ("\u26c8\ufe0f", "Thunderstorm"),
    96: ("\u26c8\ufe0f", "Thunderstorm with slight hail"),
    99: ("\u26c8\ufe0f", "Thunderstorm with heavy hail"),
}


def weather_icon(code):
    """Return (emoji, description) for a WMO weather code."""
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "\u2753", "Unknown"
    return WEATHER_CODE_MAP.get(code, ("\u2753", "Unknown"))


# ---------------------------------------------------------------------------
# WMO weather codes (used by Open-Meteo) -> emoji icon + short label.
# Using emoji instead of fetched image icons means the dashboard renders
# identically offline, in CI, and on any deployment target with zero extra
# network calls or asset hosting.
# ---------------------------------------------------------------------------
WEATHER_CODE_MAP = {
    0: ("\u2600\ufe0f", "Clear sky"),
    1: ("\U0001F324\ufe0f", "Mainly clear"),
    2: ("\u26c5", "Partly cloudy"),
    3: ("\u2601\ufe0f", "Overcast"),
    45: ("\U0001F32B\ufe0f", "Fog"),
    48: ("\U0001F32B\ufe0f", "Depositing rime fog"),
    51: ("\U0001F326\ufe0f", "Light drizzle"),
    53: ("\U0001F326\ufe0f", "Drizzle"),
    55: ("\U0001F326\ufe0f", "Dense drizzle"),
    56: ("\U0001F327\ufe0f", "Freezing drizzle"),
    57: ("\U0001F327\ufe0f", "Freezing drizzle"),
    61: ("\U0001F327\ufe0f", "Light rain"),
    63: ("\U0001F327\ufe0f", "Rain"),
    65: ("\U0001F327\ufe0f", "Heavy rain"),
    66: ("\U0001F327\ufe0f", "Freezing rain"),
    67: ("\U0001F327\ufe0f", "Freezing rain"),
    71: ("\u2744\ufe0f", "Light snow"),
    73: ("\u2744\ufe0f", "Snow"),
    75: ("\u2744\ufe0f", "Heavy snow"),
    77: ("\u2744\ufe0f", "Snow grains"),
    80: ("\U0001F326\ufe0f", "Rain showers"),
    81: ("\U0001F327\ufe0f", "Rain showers"),
    82: ("\U0001F327\ufe0f", "Violent rain showers"),
    85: ("\U0001F328\ufe0f", "Snow showers"),
    86: ("\U0001F328\ufe0f", "Heavy snow showers"),
    95: ("\u26c8\ufe0f", "Thunderstorm"),
    96: ("\u26c8\ufe0f", "Thunderstorm with hail"),
    99: ("\u26c8\ufe0f", "Thunderstorm with heavy hail"),
}


def weather_icon(code) -> tuple:
    """Return (emoji, label) for a WMO weather code. Falls back gracefully."""
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "\u2601\ufe0f", "Unknown"
    return WEATHER_CODE_MAP.get(code, ("\u2601\ufe0f", "Unknown"))
