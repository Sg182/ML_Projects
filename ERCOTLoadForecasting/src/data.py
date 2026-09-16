import pandas as pd

def load_data_model(path):
    df = pd.read_csv(path, index_col='timestamp')

    df.index = pd.to_datetime(df.index,utc=True)
    df.index = df.index.tz_convert( "America/Chicago")
    return df


if __name__ == "__main__":

    df = load_data_model( "data/processed/model_data_forecast.csv")

    print(df.shape)
    print(df.columns)