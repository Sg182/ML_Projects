import pandas as pd
import pytest

from src.features import FEATURES, select_features


def test_select_features():

    data = {feature: [1.0] for feature in FEATURES}
    data["ERCOT"] = [50000]

    df = pd.DataFrame(data)

    X = select_features(df)

    assert X.columns.tolist() == FEATURES
    assert "ERCOT" not in X.columns


def test_column_order_matches_features_list():
    # even if the input DataFrame has its columns in a different order,
    # select_features must return them in FEATURES order - training and
    # prediction both rely on this so the model always sees the same layout
    data = {feature: [1.0] for feature in reversed(FEATURES)}
    df = pd.DataFrame(data)

    X = select_features(df)

    assert X.columns.tolist() == FEATURES


def test_features_match_selected_model():
    # the saved model was trained on exactly these nine columns in this order
    # (selected in notebook 05) - if this test fails, the model in models/
    # no longer matches the code and must be retrained
    assert FEATURES == [
        "hour",
        "day_of_week",
        "month",
        "load_24h_ago",
        "load_168h_ago",
        "forecast_temperature_austin",
        "forecast_temperature_dallas",
        "forecast_temperature_houston",
        "forecast_temperature_san_antonio",
    ]


def test_missing_feature_raises():
    # a DataFrame missing a required feature should fail loudly, not silently
    data = {feature: [1.0] for feature in FEATURES}
    df = pd.DataFrame(data).drop(columns=["load_24h_ago"])

    with pytest.raises(KeyError):
        select_features(df)