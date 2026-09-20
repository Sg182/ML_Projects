import os
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.dayahead_features import build_dayahead_features

app = FastAPI()

#LOAD THE MODEL ONCE AT STARTUP, NOT ONCE PER REQUEST
dayahead_model = joblib.load("models/ercot_gb_dayahead.joblib")


class LoadInput(BaseModel):
    hour: int
    load_24h_ago: float


@app.get("/")
def home():
    return {"message": "ERCOT Load Forecasting API"}


@app.post("/echo")
def echo_load(data: LoadInput):

    return {
        "hour_received": data.hour,
        "load_received": data.load_24h_ago
    }


@app.post("/forecast/day-ahead")
def forecast_day_ahead():
    """Forecast every hour of the next operating day (notebook 06 protocol).

    Issuance is "now" in America/Chicago; the response covers all hour-ending
    timestamps of tomorrow - 23, 24, or 25 of them on daylight-saving days.

    Protocol note: the evaluated protocol fixes issuance at 10 AM, but every
    input derives from day D-1 or earlier (48h/168h lags, previous_day2
    forecasts), so the feature values are IDENTICAL for any call during day D
    once yesterday's load is fully published. Early-morning calls before that
    are rejected with 503 rather than served with substitutes. Scheduling the
    daily 10 AM call is a deployment concern, deliberately out of scope here.
    """
    #A WEB REQUEST MUST NEVER FALL INTO AN INTERACTIVE PASSWORD PROMPT,
    #SO ERCOT CREDENTIALS HAVE TO COME FROM THE ENVIRONMENT
    required = ["ERCOT_USERNAME", "ERCOT_PASSWORD", "ERCOT_SUBSCRIPTION_KEY"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"server is missing ERCOT credentials: {', '.join(missing)}",
        )

    issuance = datetime.now(ZoneInfo("America/Chicago"))

    try:
        features = build_dayahead_features(issuance)
    except RuntimeError as error:
        #REQUIRED INPUTS NOT PUBLISHED YET - REFUSE RATHER THAN INVENT VALUES
        raise HTTPException(status_code=503, detail=str(error))
    except requests.RequestException as error:
        raise HTTPException(status_code=502, detail=f"upstream data source failed: {error}")

    predictions = dayahead_model.predict(features)

    return {
        "forecast_date": str(features.index[0].date()),
        "issuance_time": issuance.isoformat(),
        "hours": len(features),
        "predictions": [
            {"timestamp": timestamp.isoformat(), "predicted_load_mw": round(float(value), 1)}
            for timestamp, value in zip(features.index, predictions)
        ],
    }
