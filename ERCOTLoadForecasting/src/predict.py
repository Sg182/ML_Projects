import joblib
import pandas as pd
from src.data import load_data_model
from src.features import select_features


def load_model(path):

    model = joblib.load(path)

    return model

import pandas as pd


def predict_one(model, feature_values):

    df = pd.DataFrame([feature_values])
    X = select_features(df)


    prediction = model.predict(X)

    return prediction[0]

def make_predictions(model, X):

    predictions = model.predict(X)

    return predictions


if __name__ == "__main__":

    model = load_model("models/ercot_gb_model.joblib")

    df = load_data_model("data/processed/model_data_forecast.csv")

    test_df = df.loc["2025-01-01":]

    X_test = select_features(test_df)

    predictions = make_predictions(
        model,
        X_test
    )

    print(predictions[:10])


if __name__ == "__main__":

    model = load_model(
        "models/ercot_gb_model.joblib"
    )

    sample = {
        "hour": 18,
        "day_of_week": 2,
        "month": 7,
        "load_24h_ago": 72000,
        "load_168h_ago": 70000,
        "forecast_temperature_houston": 36.0,
        "forecast_temperature_dallas": 38.0,
        "forecast_temperature_austin": 37.0,
        "forecast_temperature_san_antonio": 37.5,
    }

    prediction = predict_one(
        model,
        sample
    )

    print("Predicted ERCOT load:", prediction)