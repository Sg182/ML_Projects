# ERCOT Load Forecasting

[![tests](https://github.com/Sg182/ML_Projects/actions/workflows/tests.yml/badge.svg)](https://github.com/Sg182/ML_Projects/actions/workflows/tests.yml)

Day-ahead hourly electricity demand forecasting for the Texas (ERCOT) grid, under a fixed
10:00 AM issuance protocol that uses only information available at that cutoff. Independent
project. The trained model is served behind a FastAPI endpoint and reproduced from committed
code by `python -m src.train_dayahead`.

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
demand spikes. Error analysis suggested missing weather information.

An oracle-weather diagnostic (notebook 04) compared a no-weather model against one
given *realized* weather — weather that is not available at prediction time — on the
two days holding the baseline's largest errors (2025-01-06 and 2025-02-19). Daily MAE
fell 63% and 62% on those two days. Over wider windows the effect is much smaller:
−17.5% across Jan–Feb and −4.9% over the full 2025 test year. Those four figures are
XGBoost, which is what notebook 04's comparison block reports; Gradient Boosting moves the
same way but with a larger full-year gain (7.0%). Note those two days were selected
*because* they contained the worst errors, so some regression to the mean is
expected, and no quantitative definition of "cold snap" is used anywhere here.

Substituting archived day-lead GFS forecasts for realized weather (notebook 05) retained
96% and 93% of that gain on the same two days — but the same calculation gives 120% over
the full test year and 119% over Jan–Feb, because forecast weather slightly outperformed
oracle weather there and the retention ratio stops being meaningful. These are XGBoost
figures as well. Read the two-day numbers as illustrative of cold-weather behaviour,
not as a general retention rate.

The conclusion that survives all of this: forecast temperatures are good enough to be
worth including, so the production-honest model uses lag/calendar features plus day-ahead
forecast temperatures.

Connecting to ERCOT's live API then exposed that a fixed-schedule day-ahead
forecast cannot use the 24-hour lag or day-1 forecasts (they aren't available at
issuance), so notebook 06 defines a stricter 10 AM protocol using 48-hour lags
and previous-day2 GFS temperatures.

Gradient Boosting under that protocol, with development on 2022–2024 and a chronological
split (no shuffling), evaluated on 8,761 hourly observations — all 8,760 hours of calendar
2025, plus the hour-ending row closing 2024-12-31 that the `.loc["2025-01-01":]` slice
picks up:

| | Model | Naive-48h baseline | Improvement |
|---|---|---|---|
| MAE | 3,073 MW | 3,630 MW | 15.3% |
| RMSE | 3,692 MW | 5,100 MW | 27.6% |
| R² | 0.867 | 0.747 | +0.120 |

The baseline is a seasonal naive at 48 hours (same hour, two days prior) — the 24-hour
naive is inadmissible here because yesterday has not finished at a 10 AM issuance.

A multi-layer perceptron under the identical protocol (`src/train_mlp_dayahead.py`, tuned on
development data only) lands at MAE 3,092 MW, RMSE 3,680 MW, R² 0.868 — within 0.6% of
gradient boosting on MAE, and fractionally ahead of it on RMSE. That reproduces the clustering
notebook 02 found among six model classes, and is the clearest evidence that this problem is
limited by its features and its issuance protocol rather than by the function class. The MLP
is a diagnostic: it is not served and not saved to `models/`.

These numbers are **not** comparable with the rolling 24-hour results in notebooks 02–05
(e.g. RMSE 3,332 MW, R² 0.892 with no weather). That model answers a different question
under a protocol that cannot be operated on a fixed schedule; moving to the honest
protocol costs roughly 32% in MAE measured against that no-weather rolling model, or 39%
against the best rolling model (forecast weather, MAE 2,211 MW). That cost is the point
rather than an embarrassment.

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
- [x] `notebooks/04_weather_models.ipynb` — oracle-weather diagnostic (realized weather, not available at prediction time); large error reductions on the two worst days, much smaller over the full year
- [x] `notebooks/05_forecast_weather.ipynb` — leakage-free day-ahead GFS forecast temperatures; forecast weather performs close to realized weather over the test year
- [x] `notebooks/06_dayahead_protocol.ipynb` — fixed 10 AM issuance protocol: 48h lags + previous-day2 forecasts
- [x] `src/` modules reproduce the selected models bit-exactly from committed code
- [x] neural-network comparison — MLP under the day-ahead protocol, within 0.6% of gradient boosting
- [x] live feature pipeline (ERCOT real-time load + Open-Meteo forecast temperatures)
- [x] FastAPI day-ahead endpoint `POST /forecast/day-ahead` (`src/api.py`)
- [x] automated test suite — 24 test functions across `tests/`, run with `python -m pytest`
- [x] continuous integration — GitHub Actions (`.github/workflows/tests.yml`) installs from
      `requirements.txt` on a clean Ubuntu runner and runs the suite on every push and pull request
- [x] Docker — `Dockerfile` builds a reproducible local run environment
- [ ] continuous deployment — nothing is deployed; the service runs locally only
- [ ] monitoring — no metrics, logging, or drift detection in place
- [ ] rolling-model `/predict` endpoint

Known limitation: the live pipeline's load lags come from ERCOT's real-time
"actual system load" API, which reads ~2% higher (+925 MW mean in a Nov 2025
sample) than the settlement "native load" archive the models were trained on.
Timestamp/DST semantics are verified identical; the values are a different
vintage. Resolving this (e.g. retraining on API-sourced lags) is a planned
follow-up, documented in `src/train_dayahead.py`.

Later ideas: forecast apparent temperature, token caching for ERCOT auth.
