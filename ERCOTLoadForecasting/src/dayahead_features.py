"""Live feature pipeline for the day-ahead model (notebook 06 protocol).

At issuance (10 AM America/Chicago on day D) this builds one feature row per
hour-ending timestamp of operating day D+1 - 23, 24, or 25 rows depending on
daylight saving time.

Input sources, chosen so live semantics MATCH TRAINING semantics:

- Load lags: ERCOT actuals (NP6-345-CD) that are already published at fetch
  time. If a required lag hour has not been published, we raise - we never
  substitute a stale or nearby value.
- Temperatures: GFS `temperature_2m_previous_day2` from the SAME Open-Meteo
  Previous Runs API, variable and pinned model used to build the training
  data (verified live: the API serves these for future valid times, complete,
  at issuance). The "latest" forecast run is deliberately NOT used - it is
  fresher than what the model was trained on, so the reported test accuracy
  would not apply to it.
"""

import numpy as np
import requests
import pandas as pd

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.ercot_live import fetch_ercot_load, add_timestamp
from src.features import FEATURES_DAYAHEAD


CITIES = {
    "houston": (29.76, -95.36),
    "dallas": (32.78, -96.80),
    "san_antonio": (29.42, -98.49),
    "austin": (30.27, -97.74),
}


def target_hours_for_next_day(issuance_time):
    """All hour-ending timestamps of the next operating day.

    Midnight is never ambiguous (DST switches at 2 AM), so we anchor there and
    step in physical hours - DST days automatically get 23 or 25 rows.
    """
    issuance = pd.Timestamp(issuance_time).tz_convert("America/Chicago")
    next_day = issuance.date() + timedelta(days=1)
    day_after = next_day + timedelta(days=1)

    first = pd.Timestamp(f"{next_day} 00:00", tz="America/Chicago") + pd.Timedelta(hours=1)
    last = pd.Timestamp(f"{day_after} 00:00", tz="America/Chicago")

    return pd.date_range(first, last, freq="h")


def build_lag_features(load_series, target_hours):
    """48h and 168h load lags for every target hour.

    load_series only ever contains hours ERCOT has actually published, so a
    missing lag means "not available" - we raise instead of filling.
    """
    lags = {}

    for hours_back, name in [(48, "load_48h_ago"), (168, "load_168h_ago")]:
        lag_times = target_hours - pd.Timedelta(hours=hours_back)
        missing = lag_times.difference(load_series.index)

        if len(missing) > 0:
            raise RuntimeError(
                f"{name}: {len(missing)} required hours not published yet, "
                f"first missing: {missing[0]}"
            )

        lags[name] = load_series.loc[lag_times].to_numpy()

    return pd.DataFrame(lags, index=target_hours)


def fetch_forecast_temperatures(target_hours):
    """previous_day2 GFS temperatures for the target hours, one column per city.

    Same request shape as the training download (UTC, pinned gfs_global);
    raises if any target hour has no forecast value.
    """
    start = target_hours[0].tz_convert("UTC").date()
    end = target_hours[-1].tz_convert("UTC").date()

    columns = {}
    for city, (lat, lon) in CITIES.items():
        response = requests.get(
            "https://previous-runs-api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": str(start),
                "end_date": str(end),
                "hourly": "temperature_2m_previous_day2",
                "models": "gfs_global",  #SAME PINNED MODEL AS THE TRAINING DATA
                "timezone": "UTC",
            },
            timeout=60,
        )
        response.raise_for_status()
        hourly = response.json()["hourly"]

        series = pd.Series(
            hourly["temperature_2m_previous_day2"],
            index=pd.to_datetime(hourly["time"], utc=True).tz_convert("America/Chicago"),
            dtype=float,
        )
        columns[f"forecast_temperature_48h_{city}"] = series

    temps = pd.DataFrame(columns).reindex(target_hours)

    if temps.isna().any().any():
        raise RuntimeError(
            f"forecast temperatures: {int(temps.isna().sum().sum())} "
            "missing values for the target day"
        )

    return temps


def build_dayahead_features(issuance_time=None):
    """One feature row per target hour, in exact FEATURES_DAYAHEAD order.

    Note: the availability check reflects what is published AT FETCH TIME, so
    this is correct for live use (issuance = now). It is not a backtesting
    tool - passing a past issuance_time would see data published since then.
    """
    if issuance_time is None:
        issuance_time = datetime.now(ZoneInfo("America/Chicago"))

    target_hours = target_hours_for_next_day(issuance_time)

    # ERCOT actuals published so far; 9 days covers the 168h lag with margin
    issuance_day = pd.Timestamp(issuance_time).tz_convert("America/Chicago").date()
    load_df = add_timestamp(fetch_ercot_load(issuance_day - timedelta(days=9), issuance_day))

    if not load_df.index.is_unique:
        raise RuntimeError("ERCOT data contains duplicate timestamps - refusing to build lags")

    lags = build_lag_features(load_df["total"], target_hours)
    temps = fetch_forecast_temperatures(target_hours)

    features = pd.DataFrame(index=target_hours)
    features["hour"] = features.index.hour
    features["day_of_week"] = features.index.dayofweek
    features["month"] = features.index.month
    features = features.join(lags).join(temps)

    #EXACT TRAINING COLUMN ORDER - THE MODEL DEPENDS ON IT
    features = features[FEATURES_DAYAHEAD]

    # final guard: nothing missing, infinite, or otherwise unusable may reach the model
    if not np.isfinite(features.to_numpy()).all():
        raise RuntimeError("feature matrix contains missing or non-finite values")

    return features


if __name__ == "__main__":

    import joblib

    features = build_dayahead_features()

    print("target hours:", len(features), "| range:",
          features.index.min(), "to", features.index.max())
    print("\nfeatures:")
    print(features.to_string())

    #PIPELINE SMOKE TEST (NOT AN API): PREDICT TOMORROW WITH THE DAY-AHEAD MODEL
    model = joblib.load("models/ercot_gb_dayahead.joblib")
    forecast = pd.Series(model.predict(features), index=features.index, name="predicted_MW")

    print("\nday-ahead forecast for tomorrow:")
    print(forecast.round(0).to_string())
