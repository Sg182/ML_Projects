import os
import requests
import pandas as pd

from getpass import getpass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.ercot_auth import get_id_token


# NP6-345-CD: hourly actual system load by weather zone.
# Its hour-ending / DST conventions match the Native_Load Excel archive
# (verified against the 2025-11-02 transition), but its VALUES are a different
# vintage: telemetered "actual system load" reads consistently higher than the
# archive's settlement "native load" (measured +925 MW mean, +4.5 GW max, over
# a 3-day Nov 2025 sample). Live lag features therefore sit slightly above the
# training distribution - a documented train/serve skew, see train_dayahead.py.
URL = (
    "https://api.ercot.com/api/public-reports/"
    "np6-345-cd/act_sys_load_by_wzn"
)


def fetch_ercot_load(start_date, end_date):
    """Fetch hourly actual load between two operating days as a DataFrame.

    Follows the API's pagination metadata instead of assuming one page
    is enough.
    """
    # 1. Get credentials (once, reused for every page)
    token = get_id_token()

    subscription_key = os.getenv("ERCOT_SUBSCRIPTION_KEY")
    if not subscription_key:
        subscription_key = getpass("ERCOT subscription key: ")

    headers = {
        "Authorization": f"Bearer {token}",
        "Ocp-Apim-Subscription-Key": subscription_key,
    }

    # 2. Request pages until the metadata says we have them all
    all_rows = []
    page = 1

    while True:
        params = {
            "operatingDayFrom": str(start_date),
            "operatingDayTo": str(end_date),
            "size": 1000,
            "page": page,
        }

        response = requests.get(URL, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        all_rows.extend(payload["data"])

        total_pages = payload["_meta"]["totalPages"]
        if page >= total_pages:
            break
        page += 1

    # 3. Column names come with the response, in the same order as the rows
    columns = [field["name"] for field in payload["fields"]]

    return pd.DataFrame(all_rows, columns=columns)


def add_timestamp(df):
    """Build a timezone-aware hour-ending timestamp and use it as the index.

    Matches the convention of the training data: hourEnding runs
    01:00-24:00, where 24:00 is really 00:00 of the NEXT day.
    """
    df = df.copy()

    is_24 = df["hourEnding"] == "24:00"
    hour_clean = df["hourEnding"].replace("24:00", "00:00")

    naive = pd.to_datetime(df["operatingDay"] + " " + hour_clean)
    naive.loc[is_24] = naive.loc[is_24] + pd.Timedelta(days=1)

    #ON THE FALL-BACK DAY ONE CLOCK HOUR HAPPENS TWICE; DSTFlag MARKS THE SECOND
    #(STANDARD-TIME) OCCURRENCE. ambiguous=True MEANS "FIRST OCCURRENCE" TO PANDAS.
    is_second_occurrence = df["DSTFlag"].astype(str).str.upper().isin(["Y", "TRUE"])
    df["timestamp"] = naive.dt.tz_localize(
        "America/Chicago",
        ambiguous=~is_second_occurrence,
    )

    df = df.sort_values("timestamp").set_index("timestamp")
    return df


def check_hourly_coverage(df):
    """Return (missing physical hours, duplicate count) for the fetched range."""
    expected = pd.date_range(df.index.min(), df.index.max(), freq="h")
    missing = expected.difference(df.index)
    duplicates = int(df.index.duplicated().sum())
    return missing, duplicates


def get_lag_load(load_series, target_timestamp, hours_back):
    """Load exactly hours_back PHYSICAL hours before target_timestamp.

    Raises KeyError if that hour has not been published - we never
    substitute a stale or nearby value, so this lookup cannot leak.
    """
    lag_time = target_timestamp - pd.Timedelta(hours=hours_back)

    if lag_time not in load_series.index:
        raise KeyError(
            f"Load for {lag_time} is not available. "
            f"Latest published hour: {load_series.index.max()}"
        )

    return float(load_series.loc[lag_time])


if __name__ == "__main__":

    now = datetime.now(ZoneInfo("America/Chicago"))
    today = now.date()

    # 168h lag needs 7 days of history; fetch 9 to be safe
    df = fetch_ercot_load(today - timedelta(days=9), today)
    df = add_timestamp(df)

    load = df["total"]

    print("\n--- coverage ---")
    print("rows:", len(df))
    print("range:", df.index.min(), "to", df.index.max())
    missing, duplicates = check_hourly_coverage(df)
    print("missing physical hours:", len(missing))
    print("duplicate timestamps:", duplicates)

    print("\n--- publication delay ---")
    latest = df.index.max()
    print("now:                   ", now.strftime("%Y-%m-%d %H:%M %Z"))
    print("latest published hour: ", latest)
    print("delay:                 ", now - latest.to_pydatetime())

    print("\n--- availability check for a day-ahead target ---")
    # target: tomorrow at this time (next full hour), forecast issued NOW
    issuance = now
    target = (latest + pd.Timedelta(hours=24)).ceil("h") + pd.Timedelta(hours=24)

    print("forecast issuance time:", issuance.strftime("%Y-%m-%d %H:%M %Z"))
    print("target hour:           ", target)
    print("needs load at (24h):   ", target - pd.Timedelta(hours=24))
    print("needs load at (168h):  ", target - pd.Timedelta(hours=168))
    print("latest available load: ", latest)

    for hours_back in [24, 168]:
        try:
            value = get_lag_load(load, target, hours_back)
            print(f"lag {hours_back}h: {value:.1f} MW (available)")
        except KeyError as error:
            print(f"lag {hours_back}h: NOT AVAILABLE - {error}")
