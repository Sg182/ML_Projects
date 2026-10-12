# ERCOT Load Forecasting

A learning project: forecasting hourly electricity demand for the Texas (ERCOT) grid.

## Data

Historical hourly load comes from ERCOT's public "Native Load" archive
(https://www.ercot.com/gridinfo/load/load_hist) — one Excel file per year,
downloaded by the first notebook into `data/raw/` (not tracked in git).

Each row is one hour. Columns are the load in MW for ERCOT's 8 weather zones
plus the system total (`ERCOT`), timestamped with an "hour ending" convention
(01:00–24:00, local Texas time).

Quirks found while exploring the raw data:

- hours run to `24:00`, which pandas can't parse directly
- daylight saving time: each spring one hour is missing, each fall the
  repeated 02:00 hour appears twice (marked `DST`)
- one stray timestamp cell in 2022 that Excel stored as a date instead of text

## Findings so far

Baseline models performed reasonably well but failed badly during abrupt winter
demand spikes. Error analysis suggested missing weather information. Oracle-weather
experiments confirmed that weather explains much of the failure, and archived
24-hour-ahead GFS forecasts retained roughly 93–96% of the cold-snap improvement.
Therefore the final production-honest model uses lag/calendar features plus
day-ahead forecast temperatures.

Connecting to ERCOT's live API then exposed that a fixed-schedule day-ahead
forecast cannot use the 24-hour lag or day-1 forecasts (they aren't available at
issuance), so notebook 06 defines a stricter 10 AM protocol using 48-hour lags
and previous-day2 GFS temperatures — test MAE ≈ 3073 MW vs the protocol-valid
naive baseline's 3630.

## Setup

```bash
pip install -r requirements.txt
```

Run everything from the project root (data and model paths are root-relative).

**ERCOT credentials** (needed only for live data / the API endpoint): register a
free account at https://apiexplorer.ercot.com, subscribe to the Public API to get
a subscription key, then set:

```bash
export ERCOT_USERNAME="you@example.com"
export ERCOT_PASSWORD="..."
export ERCOT_SUBSCRIPTION_KEY="..."
```

Never commit these; `.env` files are gitignored. Scripts fall back to an
interactive prompt if the variables are unset — the API server does not, by design.

## Usage

```bash
python -m pytest -v                      # test suite (no credentials needed)
python -m src.train                      # retrain the rolling model
python -m src.train_dayahead             # retrain the day-ahead model
python -m src.dayahead_features          # live day-ahead pipeline (credentials)
uvicorn src.api:app                      # serve the API; POST /forecast/day-ahead
python -m scripts.smoke_dayahead_endpoint  # authenticated endpoint smoke test
python -m scripts.verify_ercot_dst       # DSTFlag semantics check vs training data
```

Regenerating ignored artifacts: `data/raw/` is downloaded by notebooks 01, 03,
05 and 06; `data/processed/` is written by notebooks 01, 03 and 05; the two
models in `models/` are rebuilt by the training commands above.

## Progress

- [x] `notebooks/01_data_exploration.ipynb` — download data (2022–2025), inspect it, clean timestamps
- [x] `notebooks/02_baseline_models.ipynb` — naive/linear/tree baselines, chronological splits, TimeSeriesSplit + GridSearchCV, error analysis
- [x] `notebooks/03_weather_features.ipynb` — hourly weather for 4 Texas metros (Open-Meteo), timezone alignment, merge with load
- [x] `notebooks/04_weather_models.ipynb` — oracle-weather diagnostic: realized weather cuts cold-snap errors ~63%
- [x] `notebooks/05_forecast_weather.ipynb` — leakage-free day-ahead GFS forecast temperatures; forecast ≈ oracle
- [x] `notebooks/06_dayahead_protocol.ipynb` — fixed 10 AM issuance protocol: 48h lags + previous-day2 forecasts
- [x] engineering: `src/` modules reproduce the selected models bit-exactly, live feature pipeline, FastAPI `POST /forecast/day-ahead`, 23 tests
- [ ] Docker, CI/CD, monitoring; rolling-model `/predict` endpoint

Known limitation: the live pipeline's load lags come from ERCOT's real-time
"actual system load" API, which reads ~2% higher (+925 MW mean in a Nov 2025
sample) than the settlement "native load" archive the models were trained on.
Timestamp/DST semantics are verified identical; the values are a different
vintage. Resolving this (e.g. retraining on API-sourced lags) is a planned
follow-up, documented in `src/train_dayahead.py`.

Later ideas: forecast apparent temperature, token caching for ERCOT auth.
