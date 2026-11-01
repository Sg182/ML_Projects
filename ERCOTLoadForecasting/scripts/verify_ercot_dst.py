"""Verify the ERCOT API's timestamp semantics against real transition records.

Run from the project root (prompts for ERCOT credentials):

    python -m scripts.verify_ercot_dst

Checks three things on the 2025-11-02 fall-back day:

1. The operating day contains 25 rows, with hourEnding 02:00 appearing twice.
2. DSTFlag marks the SECOND (standard-time) occurrence of the repeated hour.
   Decided by physics, not value-matching: load declines through the night,
   so the first (daylight-time) hour must show the higher load.
3. Every timestamp add_timestamp produces corresponds to a physical instant
   in the training data's index.

Values are deliberately NOT compared: the API serves telemetered "actual
system load", while the training archive holds settlement "native load" -
different vintages that legitimately disagree (~+925 MW mean in a Nov 2025
sample, API higher). See src/ercot_live.py and src/train_dayahead.py.
"""
import pandas as pd

from src.data import load_data_model
from src.ercot_live import fetch_ercot_load, add_timestamp

raw = fetch_ercot_load("2025-11-02", "2025-11-02")

# 1. structure of the fall-back day
print("\nrows for 2025-11-02:", len(raw), "(25 expected)")
repeated = raw[raw["hourEnding"] == "02:00"]
print("hourEnding 02:00 rows:", len(repeated), "(2 expected)")

# 2. flag semantics: the flagged row must be the LOWER-load (second) occurrence
is_flagged = repeated["DSTFlag"].astype(str).str.upper().isin(["Y", "TRUE"])
flagged_total = float(repeated[is_flagged]["total"].iloc[0])
unflagged_total = float(repeated[~is_flagged]["total"].iloc[0])
print(f"\nrepeated hour: unflagged={unflagged_total:.1f} MW, flagged={flagged_total:.1f} MW")
flag_ok = flagged_total < unflagged_total
print("flag marks the SECOND (CST) occurrence:", flag_ok)

# 3. every produced instant exists in the training index
api_df = add_timestamp(raw)
train = load_data_model("data/processed/model_data.csv")
missing = api_df.tz_convert("UTC").index.difference(train.tz_convert("UTC").index)
print("instants not found in training data:", len(missing))

ok = len(raw) == 25 and len(repeated) == 2 and flag_ok and len(missing) == 0
print("\nSEMANTICS VERIFIED" if ok else "\nMISMATCH - DO NOT TRUST add_timestamp YET")
