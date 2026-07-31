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
# Monitored locations. Started as 6 Delhi stations; expanded to a curated
# set of major Indian cities so the dashboard can show a national map.
# Open-Meteo works for any lat/lon (it's not station-based), so this list
# is just "cities we've chosen to track" — add or remove freely. Keep in
# mind: every location here costs 2 extra API calls per ingestion run
# (weather + air quality), so a few dozen is fine, but don't go wild —
# a few hundred would start stretching GitHub Actions' free run time.
# ---------------------------------------------------------------------------
LOCATIONS = {
    # Delhi NCR — original 6 stations, plus a few more real CPCB station areas
    "IGI Airport":         (28.5562, 77.1000),
    "Anand Vihar":         (28.6469, 77.3151),
    "Connaught Place":     (28.6315, 77.2167),
    "Dwarka":              (28.5921, 77.0460),
    "Okhla Phase III":     (28.5355, 77.2910),
    "Rohini":              (28.7041, 77.1025),
    "ITO":                 (28.6289, 77.2405),
    "Punjabi Bagh":        (28.6692, 77.1174),
    "R K Puram":           (28.5645, 77.1730),
    "Jahangirpuri":        (28.7286, 77.1633),

    # Major metros
    "Mumbai":              (19.0760, 72.8777),
    "Bengaluru":           (12.9716, 77.5946),
    "Chennai":             (13.0827, 80.2707),
    "Kolkata":             (22.5726, 88.3639),
    "Hyderabad":           (17.3850, 78.4867),
    "Pune":                (18.5204, 73.8567),
    "Ahmedabad":           (23.0225, 72.5714),

    # State capitals / large cities
    "Jaipur":              (26.9124, 75.7873),
    "Lucknow":             (26.8467, 80.9462),
    "Kanpur":              (26.4499, 80.3319),
    "Nagpur":              (21.1458, 79.0882),
    "Indore":              (22.7196, 75.8577),
    "Bhopal":              (23.2599, 77.4126),
    "Patna":               (25.5941, 85.1376),
    "Ludhiana":            (30.9010, 75.8573),
    "Agra":                (27.1767, 78.0081),
    "Nashik":              (19.9975, 73.7898),
    "Vadodara":            (22.3072, 73.1812),
    "Surat":               (21.1702, 72.8311),
    "Rajkot":              (22.3039, 70.8022),
    "Varanasi":            (25.3176, 82.9739),
    "Amritsar":            (31.6340, 74.8723),
    "Chandigarh":          (30.7333, 76.7794),
    "Guwahati":            (26.1445, 91.7362),
    "Bhubaneswar":         (20.2961, 85.8245),
    "Coimbatore":          (11.0168, 76.9558),
    "Kochi":               (9.9312, 76.2673),
    "Thiruvananthapuram":  (8.5241, 76.9366),
    "Visakhapatnam":       (17.6868, 83.2185),
    "Vijayawada":          (16.5062, 80.6480),
    "Ranchi":              (23.3441, 85.3096),
    "Jamshedpur":          (22.8046, 86.2029),
    "Dehradun":            (30.3165, 78.0322),
    "Shimla":              (31.1048, 77.1734),
    "Srinagar":            (34.0837, 74.7973),
    "Jodhpur":             (26.2389, 73.0243),
    "Gwalior":             (26.2183, 78.1828),
    "Raipur":              (21.2514, 81.6296),
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
# WMO weather codes (used by Open-Meteo) -> emoji icon + short label.
# Using emoji instead of fetched image icons means the dashboard renders
# identically offline, in CI, and on any deployment target with zero extra
# network calls or asset hosting/licensing to worry about.
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
