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

## Progress

- [x] `notebooks/01_data_exploration.ipynb` — download data (2022–2025), inspect it, clean timestamps
- [ ] plots of daily / weekly / seasonal load patterns
- [ ] baseline models and proper time-series train/test split

Later ideas: weather features, tree-based models, a small API for serving forecasts.
