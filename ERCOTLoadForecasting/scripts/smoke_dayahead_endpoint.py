"""One real authenticated smoke test of POST /forecast/day-ahead.

Run from the project root (prompts for ERCOT credentials; nothing is printed
or saved):

    python -m scripts.smoke_dayahead_endpoint

Calls the endpoint in-process via TestClient, so no server needs to be running.
"""
import json
import os
from getpass import getpass

if not os.getenv("ERCOT_USERNAME"):
    os.environ["ERCOT_USERNAME"] = input("ERCOT email: ")
if not os.getenv("ERCOT_PASSWORD"):
    os.environ["ERCOT_PASSWORD"] = getpass("ERCOT password: ")
if not os.getenv("ERCOT_SUBSCRIPTION_KEY"):
    os.environ["ERCOT_SUBSCRIPTION_KEY"] = getpass("ERCOT subscription key: ")

from fastapi.testclient import TestClient
import src.api as api

response = TestClient(api.app).post("/forecast/day-ahead")

print("\nstatus:", response.status_code)
body = response.json()
if response.status_code == 200:
    print("forecast_date:", body["forecast_date"])
    print("issuance_time:", body["issuance_time"])
    print("hours:", body["hours"])
    print("first 3 predictions:")
    for prediction in body["predictions"][:3]:
        print("  ", prediction)
    print("last prediction:")
    print("  ", body["predictions"][-1])
else:
    print(json.dumps(body, indent=2))
