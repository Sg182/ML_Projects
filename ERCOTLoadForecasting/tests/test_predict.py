import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.features import FEATURES
from src.predict import load_model, predict_one, make_predictions


def test_load_model_and_predict(tmp_path):
    # tiny synthetic model instead of the real ERCOT model: linear regression
    # trained on y = sum of all features, which it learns exactly - so we know
    # what the prediction SHOULD be
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((20, len(FEATURES))), columns=FEATURES)
    y = X.sum(axis=1)

    model = LinearRegression()
    model.fit(X, y)

    # save it, then load it back through the production code path
    model_path = tmp_path / "tiny_model.joblib"
    joblib.dump(model, model_path)
    loaded = load_model(model_path)

    # single prediction from a plain dict: all features = 1.0, so y should be
    # the number of features
    sample = {feature: 1.0 for feature in FEATURES}
    prediction = predict_one(loaded, sample)
    assert prediction == pytest.approx(len(FEATURES), abs=1e-6)

    # batch prediction: one value per input row, reproducing the training data
    predictions = make_predictions(loaded, X)
    assert len(predictions) == len(X)
    assert predictions == pytest.approx(y.values, abs=1e-6)


def test_predict_one_ignores_extra_keys(tmp_path):
    # extra keys in the input dict (like the target itself) must not reach the
    # model - predict_one selects exactly the FEATURES columns
    X = pd.DataFrame([[1.0] * len(FEATURES)], columns=FEATURES)
    model = LinearRegression()
    model.fit(X, [50000.0])

    sample = {feature: 1.0 for feature in FEATURES}
    sample["ERCOT"] = 99999.0  # would be target leakage if it were used

    prediction = predict_one(model, sample)
    assert prediction == pytest.approx(50000.0, abs=1e-6)
