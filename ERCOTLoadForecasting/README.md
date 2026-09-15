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

## Progress

- [x] `notebooks/01_data_exploration.ipynb` — download data (2022–2025), inspect it, clean timestamps
- [x] `notebooks/02_baseline_models.ipynb` — naive/linear/tree baselines, chronological splits, TimeSeriesSplit + GridSearchCV, error analysis
- [x] `notebooks/03_weather_features.ipynb` — hourly weather for 4 Texas metros (Open-Meteo), timezone alignment, merge with load
- [x] `notebooks/04_weather_models.ipynb` — oracle-weather diagnostic: realized weather cuts cold-snap errors ~63%
- [x] `notebooks/05_forecast_weather.ipynb` — leakage-free day-ahead GFS forecast temperatures; forecast ≈ oracle
- [ ] engineering phase: reusable modules, model persistence, inference API, tests

Later ideas: forecast apparent temperature, Docker, CI/CD, monitoring.
