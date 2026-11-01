import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.features import FEATURES, FEATURES_DAYAHEAD
from src.predict import load_model


def test_dayahead_features_match_notebook_06():
    # the day-ahead model was selected in notebook 06 on exactly these nine
    # columns in this order - if this fails, the saved day-ahead model no
    # longer matches the code and must be retrained
    assert FEATURES_DAYAHEAD == [
        "hour",
        "day_of_week",
        "month",
        "load_48h_ago",
        "load_168h_ago",
        "forecast_temperature_48h_austin",
        "forecast_temperature_48h_dallas",
        "forecast_temperature_48h_houston",
        "forecast_temperature_48h_san_antonio",
    ]


def test_rolling_and_dayahead_stay_separate():
    # the two models answer different questions and must not share features:
    # the rolling model may use yesterday's load, the day-ahead model may not
    assert "load_24h_ago" in FEATURES
    assert "load_24h_ago" not in FEATURES_DAYAHEAD
    assert "load_48h_ago" in FEATURES_DAYAHEAD
    assert "load_48h_ago" not in FEATURES


def test_dayahead_model_save_load_predict(tmp_path):
    # tiny synthetic model over the day-ahead features: y = sum of features,
    # which linear regression learns exactly, so the right answer is known
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((20, len(FEATURES_DAYAHEAD))), columns=FEATURES_DAYAHEAD)
    y = X.sum(axis=1)

    model = LinearRegression()
    model.fit(X, y)

    model_path = tmp_path / "tiny_dayahead.joblib"
    joblib.dump(model, model_path)
    loaded = load_model(model_path)

    one_row = pd.DataFrame([{feature: 1.0 for feature in FEATURES_DAYAHEAD}])
    prediction = loaded.predict(one_row[FEATURES_DAYAHEAD])

    assert len(prediction) == 1
    assert float(prediction[0]) == pytest.approx(len(FEATURES_DAYAHEAD), abs=1e-6)


def test_dayahead_missing_feature_raises():
    # selecting the day-ahead features from a frame that lacks one of them
    # must fail loudly, never silently substitute
    data = {feature: [1.0] for feature in FEATURES_DAYAHEAD}
    df = pd.DataFrame(data).drop(columns=["load_48h_ago"])

    with pytest.raises(KeyError):
        df[FEATURES_DAYAHEAD]
