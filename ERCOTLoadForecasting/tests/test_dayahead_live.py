import numpy as np
import pandas as pd
import pytest

from src.dayahead_features import target_hours_for_next_day, build_lag_features


def issuance(date_string):
    # 10 AM America/Chicago on the given day
    return pd.Timestamp(f"{date_string} 10:00", tz="America/Chicago")


def test_target_hours_normal_day():
    hours = target_hours_for_next_day(issuance("2025-06-10"))

    assert len(hours) == 24
    # hour-ending convention: first hour of the day ends at 01:00,
    # last hour ends at midnight of the day after
    assert hours[0] == pd.Timestamp("2025-06-11 01:00", tz="America/Chicago")
    assert hours[-1] == pd.Timestamp("2025-06-12 00:00", tz="America/Chicago")


def test_target_hours_spring_forward_day_has_23():
    # 2025-03-09: clocks jump 2 AM -> 3 AM, the operating day has 23 hours
    hours = target_hours_for_next_day(issuance("2025-03-08"))
    assert len(hours) == 23


def test_target_hours_fall_back_day_has_25():
    # 2025-11-02: one clock hour repeats, the operating day has 25 hours -
    # the repeated wall-clock hour must still be two DISTINCT physical instants
    hours = target_hours_for_next_day(issuance("2025-11-01"))
    assert len(hours) == 25
    assert hours.is_unique


def test_build_lag_features_values():
    # synthetic load where the value AT each hour equals its position, so the
    # 48h lag must be exactly (position - 48) - checkable without the code
    index = pd.date_range("2025-06-01", periods=300, freq="h", tz="America/Chicago")
    load = pd.Series(np.arange(300, dtype=float), index=index)

    target_hours = index[200:224]
    lags = build_lag_features(load, target_hours)

    assert list(lags.columns) == ["load_48h_ago", "load_168h_ago"]
    assert (lags["load_48h_ago"].to_numpy() == np.arange(200, 224) - 48).all()
    assert (lags["load_168h_ago"].to_numpy() == np.arange(200, 224) - 168).all()


def test_build_lag_features_missing_hour_raises():
    # remove one hour that the 48h lag needs: must raise, never fill
    index = pd.date_range("2025-06-01", periods=300, freq="h", tz="America/Chicago")
    load = pd.Series(np.arange(300, dtype=float), index=index)
    load = load.drop(index[160])  # needed by target at position 208

    target_hours = index[200:224]

    with pytest.raises(RuntimeError):
        build_lag_features(load, target_hours)
