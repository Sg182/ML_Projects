import os

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

import src.api as api
from src.dayahead_features import target_hours_for_next_day
from src.features import FEATURES_DAYAHEAD

client = TestClient(api.app)

FAKE_CREDS = {
    "ERCOT_USERNAME": "test-user",
    "ERCOT_PASSWORD": "test-password",
    "ERCOT_SUBSCRIPTION_KEY": "test-key",
}


def fake_features(issuance_date):
    # a controlled feature frame shaped exactly like the live pipeline's output,
    # so the endpoint is tested without any network calls
    issuance = pd.Timestamp(f"{issuance_date} 10:00", tz="America/Chicago")
    index = target_hours_for_next_day(issuance)

    df = pd.DataFrame(index=index)
    df["hour"] = index.hour
    df["day_of_week"] = index.dayofweek
    df["month"] = index.month
    df["load_48h_ago"] = 45000.0
    df["load_168h_ago"] = 47000.0
    for column in FEATURES_DAYAHEAD:
        if column.startswith("forecast_temperature_48h_"):
            df[column] = 30.0

    return df[FEATURES_DAYAHEAD]


def set_fake_credentials(monkeypatch):
    for name, value in FAKE_CREDS.items():
        monkeypatch.setenv(name, value)


def set_fake_model(monkeypatch):
    # a tiny stand-in model, so these tests don't need the trained artifact:
    # they check the endpoint's JSON and error handling, not model accuracy
    X = pd.DataFrame([[1.0] * len(FEATURES_DAYAHEAD)], columns=FEATURES_DAYAHEAD)
    model = LinearRegression().fit(X, [50000.0])
    monkeypatch.setattr(api, "load_dayahead_model", lambda: model)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_echo():
    response = client.post("/echo", json={"hour": 18, "load_24h_ago": 62000.0})
    assert response.status_code == 200
    assert response.json()["hour_received"] == 18


def test_dayahead_forecast_normal_day(monkeypatch):
    set_fake_credentials(monkeypatch)
    set_fake_model(monkeypatch)
    monkeypatch.setattr(api, "build_dayahead_features",
                        lambda issuance: fake_features("2025-06-10"))

    response = client.post("/forecast/day-ahead")
    assert response.status_code == 200

    body = response.json()
    assert body["forecast_date"] == "2025-06-11"
    assert body["hours"] == 24
    assert len(body["predictions"]) == 24

    first = body["predictions"][0]
    assert first["timestamp"] == "2025-06-11T01:00:00-05:00"  # timezone-aware
    assert isinstance(first["predicted_load_mw"], float)


def test_dayahead_forecast_fall_back_day_has_25_hours(monkeypatch):
    set_fake_credentials(monkeypatch)
    set_fake_model(monkeypatch)
    monkeypatch.setattr(api, "build_dayahead_features",
                        lambda issuance: fake_features("2025-11-01"))

    response = client.post("/forecast/day-ahead")
    assert response.status_code == 200
    assert response.json()["hours"] == 25


def test_dayahead_unavailable_data_returns_503(monkeypatch):
    set_fake_credentials(monkeypatch)

    def unavailable(issuance):
        raise RuntimeError("load_48h_ago: 3 required hours not published yet")

    monkeypatch.setattr(api, "build_dayahead_features", unavailable)

    response = client.post("/forecast/day-ahead")
    assert response.status_code == 503
    assert "not published" in response.json()["detail"]


def test_dayahead_missing_credentials_returns_503(monkeypatch):
    for name in FAKE_CREDS:
        monkeypatch.delenv(name, raising=False)

    response = client.post("/forecast/day-ahead")
    assert response.status_code == 503
    assert "credentials" in response.json()["detail"]


@pytest.mark.skipif(
    not os.path.exists(api.MODEL_PATH),
    reason="trained model artifact not present (regenerate: python -m src.train_dayahead)",
)
def test_real_model_artifact_loads():
    # runs locally where the artifact exists, skips on a fresh clone / in CI
    model = api.load_dayahead_model()
    assert hasattr(model, "predict")
    assert model.n_features_in_ == len(FEATURES_DAYAHEAD)
