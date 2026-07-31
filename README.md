# Delhi AQI — live 7-day forecast

A live, self-updating air quality forecasting system for six Delhi locations,
built on top of the original `notebooks/original_notebook.ipynb` analysis.
Every day it pulls fresh weather + air quality data, retrains, forecasts the
next 7 days, and serves it on a public Streamlit dashboard.

```
Live weather + AQI APIs  ->  Daily ingestion job  ->  Time-series store
        ->  Forecast model (7-day)  ->  Live dashboard
```

## What's already built

| File | What it does |
|---|---|
| `src/config.py` | Locations, feature lists, horizons, AQI color bands — single source of truth |
| `src/features.py` | Feature engineering (lags, rolling means, cyclical time features), tested and working |
| `src/ingest.py` | Pulls live data from Open-Meteo (weather + air quality), appends to archive |
| `src/train.py` | Trains one XGBoost model per location per forecast horizon (24h ... 168h) |
| `src/forecast.py` | Produces the next-7-days forecast using live weather forecasts as input |
| `app.py` | Streamlit dashboard: current conditions, AQI badge, 7-day cards, trend charts |
| `.github/workflows/ingest.yml` | Runs `ingest.py` daily via GitHub Actions (free) |
| `.github/workflows/retrain.yml` | Runs `train.py` + `forecast.py` daily |

`src/features.py` has been tested end-to-end with synthetic data (feature
shapes, training matrix, inference row alignment all verified). **The API
calls in `ingest.py` and `forecast.py` have not been tested against the live
internet** — I built this in a sandboxed environment with no network access,
so those are written carefully against Open-Meteo's documented response
format but need a real run to confirm. That's step 1 below.

## Why this design

- **Open-Meteo** for both weather and air quality: completely free, no API
  key, and its air quality endpoint returns a `us_aqi` field, used here as
  the `aqi_index` target. Swap in a CPCB/WAQI feed later if you want the
  official Indian AQI scale instead — `config.py` and `ingest.py` are the
  only two files that would need to change.
- **Direct multi-horizon models** (7 separate models per location, one per
  24h step) instead of recursively feeding a model its own predictions.
  Direct models don't compound error over the week, at the cost of
  training 7x more models — cheap for XGBoost.
- **Full history kept, dashboard shows 30 days.** `history.parquet` is
  never trimmed (needed for the model to learn winter stubble-burning /
  Diwali spikes vs. monsoon lows); `last_30_days.parquet` is the small
  slice the dashboard actually reads.
- **One-hot text encoding of weather condition dropped.** The original
  notebook one-hot encoded `condition_text`/`description` strings, which is
  fragile for a live pipeline (the model would break if a live API ever
  returns a weather description string it hasn't seen). The live version
  uses the numeric WMO `weather_code` directly instead.
- **Weather icons via emoji, not fetched images.** The dashboard maps
  Open-Meteo's WMO weather codes to emoji (`src/config.py:WEATHER_CODE_MAP`)
  instead of downloading icon assets from a CDN. Zero extra network calls,
  no broken-image risk, identical rendering everywhere Streamlit runs.
- **Forecast guardrail.** `forecast.py` clamps predictions to a generous
  multiple of the recent observed AQI range, since with only a few weeks of
  training data the model can extrapolate to implausible values. Loosen or
  remove this once you have a few months of history and trust the model's
  own range more. The dashboard also shows a banner while history is under
  ~45 days, so forecasts aren't presented as more trustworthy than they are.

## Setup — do this first

1. **Create a new GitHub repo** and push everything in this zip to it.

2. **Backfill history locally** (one-time, needs internet):
   ```bash
   pip install -r requirements.txt
   python src/ingest.py --past-days 30
   ```
   This seeds `data/history.parquet` with ~30 days of hourly data per
   location so there's enough for lag/rolling features and a first
   training run. If it errors, check the traceback against Open-Meteo's
   current API docs (https://open-meteo.com/en/docs) — endpoints
   occasionally add/rename parameters.

3. **Train and forecast locally**, to confirm it all works before automating:
   ```bash
   python src/train.py
   python src/forecast.py
   ```
   Check `models/training_report.json` for MAE per location/horizon.

4. **Run the dashboard locally**:
   ```bash
   streamlit run app.py
   ```

5. **Commit the seeded `data/` and `models/` folders** so the repo has a
   working baseline before automation takes over:
   ```bash
   git add data/ models/
   git commit -m "Initial backfill + trained models"
   git push
   ```

6. **Enable GitHub Actions**: the two workflows in `.github/workflows/`
   will start running on their schedule automatically once pushed (they
   also have `workflow_dispatch`, so you can trigger them manually from
   the Actions tab to test immediately rather than waiting for the cron).

7. **Deploy the dashboard**: go to
   [share.streamlit.io](https://share.streamlit.io), connect your GitHub
   repo, point it at `app.py`. You get a free public URL
   (`yourname-delhi-aqi.streamlit.app`) that redeploys automatically
   whenever the repo updates — including the daily bot commits from
   GitHub Actions, so the live site refreshes itself every day.

## Things to check / likely first bugs

- **Open-Meteo response shape.** I wrote `ingest.py`/`forecast.py` against
  the documented hourly JSON format, but live APIs drift — if `fetch_*`
  throws a `KeyError`, print the raw JSON response and adjust the
  `rename()` mapping.
- **`MIN_ROWS_REQUIRED` in `train.py`** is set to 200 rows — with 6
  locations and hourly data, 30 days gives ~720 rows per location, so
  this should pass, but the `168` (7-day rolling) and `168`-hour horizon
  features eat into that with `dropna()`. If a location has too few rows
  after backfill, just wait a few more days of ingestion or increase
  `--past-days` (Open-Meteo supports up to ~92 days of history this way).
- **GitHub Actions committing back to the repo** needs the default
  `GITHUB_TOKEN` to have write permission — if the push step fails, go to
  repo Settings → Actions → General → Workflow permissions → "Read and
  write permissions".

## Further development ideas

- **Model monitoring**: log daily MAE per location to a small CSV and
  chart it on the dashboard — catches model drift over time.
- **Alerts**: a GitHub Actions step that pings a Slack/Discord webhook
  when a forecast crosses into "Very unhealthy" or worse.
- **Better AQI scale**: swap Open-Meteo's `us_aqi` for a live CPCB/WAQI
  feed to match India's official AQI bands, which differ from the US
  scale in a few pollutant breakpoints.
- **Confidence intervals**: train XGBoost with quantile loss (or a
  separate model per quantile) to show a forecast range, not just a
  point estimate.
- **Map view**: `st.map()` or Plotly's mapbox scatter with all 6
  locations colored by current AQI, for an at-a-glance city view.
- **Caching/cost**: `st.cache_data(ttl=1800)` already limits re-reads;
  if you add more locations, consider moving `history.parquet` to a
  free-tier Postgres (Supabase) instead of committing large Parquet
  files to git.

## Portfolio write-up checklist

- [ ] Link the live Streamlit URL at the top of this README
- [ ] Add 2-3 dashboard screenshots
- [ ] Note the MAE/R2 achieved per location (from `training_report.json`)
- [ ] Explain the direct multi-horizon design choice (this README's
      "Why this design" section is a good starting draft)
- [ ] Link back to `notebooks/original_notebook.ipynb` as the EDA/prototype
      that the live pipeline is based on
