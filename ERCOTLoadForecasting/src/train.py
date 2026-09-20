from src.data import load_data_model
from src.features import select_features
import os
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def train_model():
    # 1. Load the dataset (the 492-hour weather-archive gap was already dropped
    #    when notebook 05 saved this file, so no gap handling is needed here)
    df = load_data_model("data/processed/model_data_forecast.csv")

    # 2. Chronological split: develop on 2022-2024, test on 2025 - never random
    dev_df = df.loc[:"2024-12-31"]
    test_df = df.loc["2025-01-01":]

    # 3. Select features and targets
    X_dev = select_features(dev_df)
    y_dev = dev_df["ERCOT"]

    X_test = select_features(test_df)
    y_test = test_df["ERCOT"]

    # 4. Train the model
    #HYPERPARAMETERS WERE SELECTED IN NOTEBOOK 05 BY GRID SEARCH CV WITH
    #TIMESERIESSPLIT ON DEVELOPMENT DATA ONLY - DO NOT RETUNE THEM HERE
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=14,
        random_state=42,
    )

    model.fit(X_dev, y_dev)

    # 5. Evaluate
    predictions = model.predict(X_test)

    print("MAE:", mean_absolute_error(y_test, predictions))
    print("RMSE:", root_mean_squared_error(y_test, predictions))
    print("R2:", r2_score(y_test, predictions))

    model_path = "models/ercot_gb_model.joblib"

    #SAVES THE TRAINED MODEL USING joblib
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, model_path)
    print("Model saved to:", model_path)

    return model

if __name__ == "__main__":
    train_model()