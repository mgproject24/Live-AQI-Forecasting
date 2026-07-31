# Delhi AQI — Live 7-Day Forecast

A self-updating air quality forecasting system for six locations across Delhi. A daily automated pipeline ingests live weather and air quality data, retrains a forecasting model, and publishes a 7-day AQI forecast to a public dashboard — with no manual intervention required.

**Live demo:** [add your Streamlit Cloud URL here]
**Source notebook (original EDA/modeling):** `notebooks/original_notebook.ipynb`

---

## Table of contents

- [Overview](#overview)
- [Screenshots](#screenshots)
- [Architecture](#architecture)
- [Methodology](#methodology)
- [Results](#results)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Automation](#automation)
- [Deployment](#deployment)
- [Known limitations](#known-limitations)
- [Future work](#future-work)
- [Data sources](#data-sources)

---

## Overview

Most air quality forecasting projects end at a static notebook trained once on historical data. This project instead runs as a continuously operating system:

- A scheduled job fetches live weather and air quality data every day
- History accumulates in a versioned data store rather than being discarded
- A forecasting model retrains daily on the growing dataset
- A public dashboard displays current conditions and a 7-day forecast per location, refreshing automatically as new data and predictions land

The goal was to practice the full lifecycle of a forecasting product — data engineering, model design, automation, and deployment — rather than only the modeling step.

## Screenshots

<!-- Add your own screenshots here, e.g.: -->
<!-- ![Dashboard overview](assets/dashboard-overview.png) -->
<!-- ![7-day forecast cards](assets/forecast-cards.png) -->

Add screenshots of the running dashboard to the `assets/` folder and reference them above.

## Architecture

```
Live weather + air quality APIs (Open-Meteo)
            │
            ▼
   Daily ingestion job (GitHub Actions)
            │
            ▼
   Time-series data store (Parquet)
            │
            ▼
   Forecast models — one XGBoost model
   per location per horizon (24h–168h)
            │
            ▼
      Live dashboard (Streamlit)
```

<!-- Optionally replace the block above with an image, e.g.: -->
<!-- ![Architecture diagram](assets/architecture-diagram.png) -->

Two independent scheduled workflows keep the system current:

| Workflow | Schedule | Responsibility |
|---|---|---|
| `ingest.yml` | Daily | Fetch latest weather + air quality data, append to the archive |
| `retrain.yml` | Daily | Retrain all models on updated history, regenerate the 7-day forecast |

Both commit their output back to the repository, which triggers an automatic redeploy of the connected Streamlit app — so the public dashboard reflects new data without any manual step.

## Methodology

**Forecast target.** AQI (US EPA scale, as returned by Open-Meteo's air quality API) for six Delhi locations, at 1-hour resolution.

**Modeling approach: direct multi-horizon forecasting.** Rather than a single model that predicts one step ahead and feeds its own output back in recursively — which compounds error over a 7-day horizon — this project trains **seven independent models per location**, one for each horizon (24h, 48h, 72h, 96h, 120h, 144h, 168h). Each model is trained to predict AQI at `t + horizon` directly from features known at time `t`, plus the forecasted weather for `t + horizon` (using Open-Meteo's weather forecast as a proxy input, since it is available at inference time).

**Features:**
- Lag features (1h, 24h) and rolling means (24h, 168h) for AQI, key pollutants, and weather variables
- Cyclical time encodings (hour-of-day, month-of-year) to capture diurnal and seasonal patterns
- Forecasted weather (temperature, humidity, pressure, wind, weather code) at the target horizon

**Model:** XGBoost regression, one model per (location, horizon) pair.

**Evaluation:** time-ordered train/test split (no shuffling, to avoid leakage across the temporal structure); MAE and R² reported per location and horizon in `models/training_report.json` after each training run.

## Results

<!-- Fill this in from your own models/training_report.json once you have
     a few weeks of accumulated history. Example structure: -->

| Location | Horizon | MAE | R² |
|---|---|---|---|
| IGI Airport | 24h | — | — |
| IGI Airport | 168h | — | — |
| ... | ... | ... | ... |

*Add a short note here on how these compare to a naive baseline (e.g. "AQI in 24h = AQI now"), once implemented.*

## Tech stack

- **Data / modeling:** Python, pandas, XGBoost, scikit-learn
- **Ingestion:** Open-Meteo APIs (weather + air quality), requests
- **Storage:** Parquet (full history + rolling 30-day slice for the dashboard)
- **Automation:** GitHub Actions (scheduled workflows)
- **Dashboard:** Streamlit, Plotly
- **Deployment:** Streamlit Community Cloud

## Project structure

```
delhi-aqi-live/
├── data/                     # history.parquet, last_30_days.parquet, forecast.parquet
├── src/
│   ├── config.py              # locations, feature config, AQI bands
│   ├── features.py            # feature engineering, shared by training and inference
│   ├── ingest.py               # live data ingestion
│   ├── train.py                 # per-location, per-horizon model training
│   └── forecast.py              # 7-day forecast generation
├── models/                    # trained model artifacts + training_report.json
├── app.py                      # Streamlit dashboard
├── .github/workflows/          # daily ingestion and retraining automation
├── notebooks/                  # original exploratory notebook
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt

# One-time backfill of history (30+ days recommended)
python src/ingest.py --past-days 30

# Train models and generate the first forecast
python src/train.py
python src/forecast.py

# Run the dashboard locally
streamlit run app.py
```

## Automation

The pipeline runs unattended once deployed:

1. Push the repository to GitHub.
2. Under **Settings → Actions → General → Workflow permissions**, select **"Read and write permissions"** so the scheduled jobs can commit updated data back to the repo.
3. The two workflows in `.github/workflows/` run on their own daily schedule (and can also be triggered manually from the **Actions** tab via `workflow_dispatch`).
4. Each run commits its output — new data, retrained models, or an updated forecast — back to the repository.

## Deployment

1. Go to [share.streamlit.io](https://share.streamlit.io) and connect the GitHub repository.
2. Point the app at `app.py`.
3. Deploy — the app receives a public URL and automatically redeploys on every push, including the daily commits from GitHub Actions.

## Known limitations

- **Modeled, not measured, air quality data.** Open-Meteo's air quality data is a model estimate (Copernicus CAMS), not a live reading from a physical monitoring station, so values will differ from station-based sources such as CPCB or AQICN. A future iteration could swap in a real per-station API.
- **Forecast reliability depends on accumulated history.** With only a few weeks of data, longer-horizon forecasts (120h–168h) are prone to unrealistic extrapolation, since the model has seen very few complete weekly or seasonal cycles. A sanity clamp in `forecast.py` currently bounds predictions to a plausible range as a temporary safeguard; this should be loosened once more history accumulates and the model's own extrapolation can be trusted.
- **No confidence intervals.** Forecasts are currently point estimates only.

## Future work

- Baseline comparison (naive/seasonal persistence) to quantify model value over a trivial forecast
- Walk-forward backtesting across multiple time origins, rather than a single train/test split
- Feature importance / SHAP analysis
- Real per-station air quality data source (e.g. WAQI/CPCB) as an alternative to the modeled Open-Meteo values
- Quantile regression for forecast confidence intervals
- Automated model-drift monitoring (tracking MAE over time)

## Data sources

- [Open-Meteo Weather API](https://open-meteo.com/en/docs) — historical and forecast weather data
- [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) — historical and current air quality data
