#ROLLING MODEL (NOTEBOOK 05): SAME NAMES AND ORDER AS SELECTED - DO NOT REORDER
FEATURES = [
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

#DAY-AHEAD MODEL (NOTEBOOK 06, 10 AM ISSUANCE PROTOCOL): ONLY INFORMATION
#AVAILABLE BEFORE 10 AM ON THE DAY BEFORE THE TARGET - DO NOT REORDER
FEATURES_DAYAHEAD = [
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


def select_features(df):
    X = df[FEATURES]
    return X


if __name__ == "__main__":

    from src.data import load_data_model

    df = load_data_model(
        "data/processed/model_data_forecast.csv"
    )

    X = select_features(df)
    y = df["ERCOT"]

    print("Full dataset:", df.shape)
    print("Features:", X.shape)
    print("Target:", y.shape)

    print("\nSelected features:")
    print(X.columns.tolist())