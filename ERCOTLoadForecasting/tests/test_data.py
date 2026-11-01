import pandas as pd

from src.data import load_data_model


def test_load_data_model(tmp_path):
    # small fake CSV with the same timestamp format as the real processed data:
    # tz-aware local time whose UTC offset changes with daylight saving
    # (-06:00 in winter, -05:00 in summer) - the reason load_data_model
    # parses to UTC first instead of using parse_dates
    csv_path = tmp_path / "small_model_data.csv"
    csv_path.write_text(
        "timestamp,ERCOT\n"
        "2024-01-15 01:00:00-06:00,40000\n"
        "2024-01-15 02:00:00-06:00,41000\n"
        "2024-07-15 01:00:00-05:00,60000\n"
    )

    df = load_data_model(csv_path)

    # timestamp became the index, as a proper timezone-aware DatetimeIndex
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "America/Chicago"

    # no rows lost, values intact
    assert len(df) == 3
    assert df["ERCOT"].tolist() == [40000, 41000, 60000]

    # local clock time is preserved (1am Chicago stays 1am, winter and summer)
    assert df.index[0].hour == 1
    assert df.index[2].hour == 1


def test_load_data_model_date_slicing(tmp_path):
    # date-string slicing must work on the loaded index - this silently broke
    # in notebook 02 when the index was left as strings
    csv_path = tmp_path / "small_model_data.csv"
    csv_path.write_text(
        "timestamp,ERCOT\n"
        "2024-12-31 22:00:00-06:00,45000\n"
        "2024-12-31 23:00:00-06:00,44000\n"
        "2025-01-01 00:00:00-06:00,43000\n"
    )

    df = load_data_model(csv_path)

    # slicing up to a date must INCLUDE that whole day
    dev = df.loc[:"2024-12-31"]
    test = df.loc["2025-01-01":]

    assert len(dev) == 2
    assert len(test) == 1
    assert len(dev) + len(test) == len(df)
