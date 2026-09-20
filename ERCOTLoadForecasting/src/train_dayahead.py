"""Train the day-ahead ERCOT load model (notebook 06 protocol).

DATA-AVAILABILITY CONTRACT
--------------------------
Issuance: 10:00 AM America/Chicago on day D, forecasting every hour-ending
timestamp of operating day D+1. Because of daylight saving time, an operating
day can contain 23, 24, or 25 physical hours - never assume 24.

Every feature must be available before issuance:

- load_48h_ago / load_168h_ago: ERCOT actuals (report NP6-345-CD). All of day
  D-1 must be published by 10 AM on D. Our live inspection measured a ~1-2 hour
  publication delay AT ONE POINT IN TIME - that is an observation, not a
  guaranteed service level, so the live pipeline must re-check availability at
  every issuance and refuse to predict with missing lags.
- forecast_temperature_48h_* : GFS "previous_day2" temperatures (the forecast
  issued during day D-1, i.e. always before issuance). The training data comes
  from Open-Meteo's previous-runs ARCHIVE; a live forecast API is not
  guaranteed to reproduce those exact values, so the forecast source and its
  issuance-time semantics must be verified before live integration.

KNOWN TRAIN/SERVE SKEW: the training lags come from ERCOT's Native Load Excel
archive (settlement data), but the live pipeline's lags come from the NP6-345-CD
API (telemetered actual system load), which reads consistently HIGHER - measured
+925 MW mean, +4.5 GW max over a 3-day Nov 2025 sample. Timestamp semantics were
verified identical; the values are a different vintage. Live predictions are
therefore built from slightly out-of-distribution lag inputs. Resolving this
(retraining on API-sourced lags, or finding a native-load API product) is a
deliberate future modeling decision, not something to patch silently.
"""

import os
import joblib
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.data import load_data_model
from src.features import FEATURES_DAYAHEAD


def build_dayahead_dataset():
    """Rebuild notebook 06's merged dataset from the stored data files.

    (Notebook 06 built this in memory and never saved a CSV, so the same
    three-step merge lives here, in code, where it stays reproducible.)
    """
    # 1. Base: notebook 05's processed dataset (calendar features, 168h lag, target)
    df = load_data_model("data/processed/model_data_forecast.csv")

    # 2. 48h load lag, computed from the CONTINUOUS load history - the forecast
    #    dataset has a ~490-hour archive gap where a row shift would silently
    #    pick the wrong hours
    load_history = load_data_model("data/processed/model_data.csv")
    hour_steps = load_history.index.to_series().diff().dropna()
    assert (hour_steps == pd.Timedelta(hours=1)).all()
    load_48h_ago = load_history["ERCOT"].shift(48).rename("load_48h_ago")

    # 3. 48h-lead (previous_day2) GFS temperatures, downloaded by notebook 06
    d2 = pd.read_csv("data/raw/forecast_weather_d2_4cities.csv")
    d2["time"] = pd.to_datetime(d2["time"], utc=True).dt.tz_convert("America/Chicago")
    d2_wide = d2.pivot(index="time", columns="city", values="temperature_2m_previous_day2")
    d2_wide.columns = [f"forecast_temperature_48h_{city}" for city in d2_wide.columns]

    df = df.join(load_48h_ago, how="left")
    df = df.join(d2_wide, how="left")

    # drop the few hours (0.2%) with no archived forecast or lag; the 2025
    # test year is complete, so this only touches development data
    new_columns = ["load_48h_ago"] + list(d2_wide.columns)
    df = df.dropna(subset=new_columns)

    return df


def train_dayahead_model():
    df = build_dayahead_dataset()

    #SAME CHRONOLOGICAL SPLIT AS NOTEBOOK 06 - NO RANDOM SPLITTING
    dev_df = df.loc[:"2024-12-31"]
    test_df = df.loc["2025-01-01":]

    X_dev = dev_df[FEATURES_DAYAHEAD]
    y_dev = dev_df["ERCOT"]

    X_test = test_df[FEATURES_DAYAHEAD]
    y_test = test_df["ERCOT"]

    #HYPERPARAMETERS SELECTED IN NOTEBOOK 06 BY GRIDSEARCHCV WITH TIMESERIESSPLIT
    #ON DEVELOPMENT DATA ONLY - DO NOT RETUNE THEM HERE
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=5,
        random_state=42,
    )

    model.fit(X_dev, y_dev)

    predictions = model.predict(X_test)

    print("MAE:", mean_absolute_error(y_test, predictions))
    print("RMSE:", root_mean_squared_error(y_test, predictions))
    print("R2:", r2_score(y_test, predictions))

    model_path = "models/ercot_gb_dayahead.joblib"
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_path)

    print("Model saved to:", model_path)

    return model


if __name__ == "__main__":
    train_dayahead_model()
