# India AQI — Live 7-Day Forecast

A self-updating air quality forecasting system covering 48 locations across India (originally scoped to 6 Delhi stations, expanded to a nationwide map). A daily automated pipeline ingests live weather and air quality data, retrains a forecasting model, and publishes a 7-day AQI forecast per location on a public dashboard — with no manual intervention required.

**Live demo:** (https://live-aqi-forecasting-qwmfvuvaad96qqsqcaguo9.streamlit.app/)
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
- [MLOps and reliability](#mlops-and-reliability)
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
- **Diwali proximity** (`days_to_diwali`, `is_diwali_window`) — firecracker use around Diwali is one of the most well-documented short-term AQI spikes in North India, and it's a far more direct signal than hoping month/day-of-year features indirectly capture it. Diwali follows the Hindu lunisolar calendar, so its date shifts year to year rather than falling on a fixed day — the dates used (2024–2028) were verified via web search rather than computed or guessed; see `src/festivals.py` for sources and instructions to extend the table for later years.

**Model:** XGBoost regression, one model per (location, horizon) pair. Hyperparameters default to a fixed configuration, but can be tuned per (location, horizon) by `tune.py` — see [MLOps and reliability](#mlops-and-reliability).

**Evaluation: walk-forward backtesting against two baselines.** A single train/test split on time-series data produces one noisy estimate that depends heavily on which slice landed in the test set. Instead, `train.py` runs a 5-fold walk-forward backtest per (location, horizon): each fold trains only on data before its test window and evaluates only on data after it, sliding forward through time, and the reported metrics are averaged across folds.

Every fold is scored against two baselines computed on the exact same rows:
- **Naive persistence** — "AQI in N hours = AQI right now"
- **Weekly seasonal** — "AQI at time X = AQI at the same time exactly 7 days earlier" (this reaches back in time relative to *now*, never forward, so it's a fair comparison — not a hidden peek at the future)

A model is only deployed (see [MLOps and reliability](#mlops-and-reliability)) if it beats the better of these two baselines. MAE, RMSE, and MAPE are all reported, not just MAE alone.

**Prediction intervals via conformal prediction.** Every forecast comes with an 80%/90% prediction interval, not just a point estimate. The interval half-width is calibrated from the empirical quantiles of the walk-forward backtest's pooled out-of-sample residuals — a deliberately simple form of split conformal prediction (it assumes residual magnitude doesn't vary much across the prediction range, i.e. no bucketing by predicted value). It's a real simplification, not textbook-complete conformal prediction, but it's meaningfully more honest than a point forecast with no uncertainty at all. See `train.py`'s `_conformal_intervals`.

## Results

<!-- Fill this in from your own models/training_report.json once you have
     a few weeks of accumulated history. Example structure: -->

| Location | Horizon | Model MAE | Naive baseline MAE | Seasonal baseline MAE | RMSE | MAPE |
|---|---|---|---|---|---|---|
| IGI Airport | 24h | — | — | — | — | — |
| IGI Airport | 168h | — | — | — | — | — |
| ... | ... | ... | ... | ... | ... | ... |

If a location/horizon's model MAE isn't clearly better than both baseline columns, that's not a model worth trusting yet — `train.py`'s promotion gate already enforces this before deployment, but it's worth stating plainly here too.

## Tech stack

- **Data / modeling:** Python, pandas, XGBoost, scikit-learn
- **Ingestion:** Open-Meteo APIs (weather + air quality), requests
- **Storage:** Parquet (full history + rolling 30-day slice for the dashboard)
- **Automation:** GitHub Actions (scheduled workflows)
- **Testing / CI:** pytest, Streamlit's `AppTest` headless test harness
- **Dashboard:** Streamlit, Plotly
- **Deployment:** Streamlit Community Cloud

## Project structure

```
delhi-aqi-live/
├── data/                     # history.parquet, last_30_days.parquet, forecast.parquet
├── src/
│   ├── config.py              # locations, feature config, AQI bands
│   ├── features.py            # feature engineering, shared by training and inference
│   ├── festivals.py            # Diwali calendar feature (verified dates, see docstring)
│   ├── evaluation.py            # shared walk-forward backtest + metrics (train.py and tune.py both use this)
│   ├── ingest.py               # live data ingestion
│   ├── train.py                 # per-location, per-horizon training + promotion gate + conformal intervals
│   ├── tune.py                   # periodic (weekly) hyperparameter grid search
│   ├── forecast.py              # 7-day forecast generation, incl. prediction intervals
│   └── monitor.py               # drift / repeated-rejection detection over the experiment log
├── tests/
│   ├── test_features.py        # unit tests, incl. regression tests for two real production bugs
│   ├── test_evaluation.py       # metrics tests (pure) + backtest test (skipped if xgboost absent)
│   └── test_app_smoke.py        # headless dashboard smoke test (Streamlit AppTest)
├── models/                    # trained model artifacts, training_report.json, experiment_log.jsonl, best_params.json
├── app.py                      # Streamlit dashboard
├── .github/workflows/
│   ├── ci.yml                   # runs the test suite on every push/PR
│   ├── ingest.yml                # daily data ingestion
│   ├── retrain.yml               # daily retrain + forecast + drift monitor + alerting
│   └── tune.yml                   # weekly hyperparameter tuning (kept separate - see tune.py docstring)
├── notebooks/                  # original exploratory notebook
├── requirements.txt
└── requirements-dev.txt         # adds pytest for local/CI test runs
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

# Run the test suite
pip install -r requirements-dev.txt
pytest tests/ -v
```

## Automation

The pipeline runs unattended once deployed:

1. Push the repository to GitHub.
2. Under **Settings → Actions → General → Workflow permissions**, select **"Read and write permissions"** so the scheduled jobs can commit updated data back to the repo.
3. The workflows in `.github/workflows/` each run on their own schedule (and can also be triggered manually from the **Actions** tab via `workflow_dispatch`): data ingestion and retraining run daily, hyperparameter tuning runs weekly.
4. Each run commits its output — new data, retrained models, tuned hyperparameters, or an updated forecast — back to the repository.

## MLOps and reliability

Four practices keep this pipeline from silently degrading or shipping a broken dashboard:

**1. CI on every push.** `tests/` covers feature engineering (`test_features.py`), the shared evaluation logic (`test_evaluation.py`), and the dashboard itself (`test_app_smoke.py`, using Streamlit's `AppTest` to run the full app headlessly and fail if anything raises). `.github/workflows/ci.yml` runs this suite on every push and pull request.

  *Why this exists:* a Plotly API deprecation (`scatter_mapbox` → `scatter_map`) broke the live map in production before this test suite existed — the dashboard crashed on Streamlit Cloud with the underlying error redacted, and it wasn't caught until a user reported it. `test_app_smoke.py` is written specifically so that class of failure — a dependency change breaking a page that "worked when I last checked" — fails CI instead of reaching production.

**2. Promotion gating on every retrain.** `train.py` no longer blindly overwrites a deployed model. A freshly trained model is only promoted if it (a) beats the better of the naive-persistence and weekly-seasonal baselines on a 5-fold walk-forward backtest, and (b) isn't meaningfully worse than the model currently in production. A daily automated retrain with no gate can quietly regress — this makes a single bad retrain unable to take down a working model.

**3. Drift monitoring across runs.** Every training run — promoted or not — is appended to `models/experiment_log.jsonl`. `monitor.py` reads that history and flags two patterns the per-run gate can't see on its own: cumulative drift (MAE creeping up by, say, 9% a day forever, never large enough in a single step to trip the gate, but +80% over two weeks) and repeated rejections (a model that has stopped beating its baseline for several runs in a row, suggesting a data or feature problem rather than noise). `retrain.yml` runs this after every retrain and opens (or updates) a GitHub issue automatically if it finds anything.

**4. Hyperparameter tuning kept separate from the daily retrain.** `tune.py` runs a small grid search per (location, horizon), evaluated with the same walk-forward backtest used everywhere else, and writes the winning parameters to `models/best_params.json` for `train.py` to pick up. This runs weekly (`tune.yml`), not daily — grid search multiplies training cost by the grid size, and hyperparameters that were good last week are still almost certainly fine today, so there's no reason to pay that cost every single day.

## Deployment

1. Go to [share.streamlit.io](https://share.streamlit.io) and connect the GitHub repository.
2. Point the app at `app.py`.
3. Deploy — the app receives a public URL and automatically redeploys on every push, including the daily commits from GitHub Actions.

## Known limitations

- **Modeled, not measured, air quality data.** Open-Meteo's air quality data is a model estimate (Copernicus CAMS), not a live reading from a physical monitoring station, so values will differ from station-based sources such as CPCB or AQICN. A future iteration could swap in a real per-station API.
- **Forecast reliability depends on accumulated history.** With only a few weeks of data, longer-horizon forecasts (120h–168h) are prone to unrealistic extrapolation, since the model has seen very few complete weekly or seasonal cycles. A sanity clamp in `forecast.py` currently bounds predictions to a plausible range as a temporary safeguard; this should be loosened once more history accumulates and the model's own extrapolation can be trusted.
- **The walk-forward backtest needs enough history to run at all.** With under a few hundred rows per location, `train.py` will skip a horizon entirely rather than report an unreliable backtest — this is intentional (see `MIN_ROWS_REQUIRED` and the fold-size check in `walk_forward_backtest`), but it means new locations take longer to get their first deployed model than the old single-split approach did.
  - *(An earlier version of this logic had a real bug here: right at the boundary — data volume just above `MIN_ROWS_REQUIRED` — the backtest would silently run only 1 fold instead of 5, with no warning, because the first fold's training window wasn't guaranteed to meet the minimum. Found by testing across a range of data volumes rather than just one, fixed by flooring the initial training window at `MIN_ROWS_REQUIRED` and logging explicitly whenever fewer folds run than requested.)*
- **The prediction intervals are a simplified form of conformal prediction.** The interval half-width is a single fixed value per (location, horizon), from the pooled backtest residual quantiles — it doesn't account for the interval potentially needing to be wider when the model is predicting an unusual/extreme value versus a typical one. A fuller implementation would bucket residuals by predicted-value range.
- **The Diwali calendar only covers 2024–2028.** Outside that range, `days_to_diwali` falls back to the nearest listed year, which becomes progressively less accurate for years further from that window. See `src/festivals.py` for how to extend it.
- **The hyperparameter grid in `tune.py` is small and hand-picked**, not a systematic search (e.g. Optuna/Bayesian optimization) — a deliberate tradeoff to avoid a new dependency that wasn't installable in the environment this was built in, not a claim that it's the optimal search strategy.

## Future work

- Feature importance / SHAP analysis
- Real per-station air quality data source (e.g. WAQI/CPCB) as an alternative to the modeled Open-Meteo values
- Compare XGBoost against LightGBM/CatBoost as alternative model families
- NASA FIRMS fire-count data as a direct stubble-burning-season signal (requires registering for a free API key)
- Cross-city spatial features (pollution transport between nearby locations)
- Bucketed/quantile-regression-based intervals instead of the current single-fixed-width conformal approach
- Experiment tracking beyond the current JSONL log (e.g. MLflow) if the project outgrows a flat file
- Optuna or similar if the hand-picked grid in `tune.py` stops being good enough

## Data sources

- [Open-Meteo Weather API](https://open-meteo.com/en/docs) — historical and forecast weather data
- [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) — historical and current air quality data
