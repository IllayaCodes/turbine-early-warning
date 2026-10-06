import pandas as pd
import pytest

from src.turbine_ew.split import time_split

SPLIT = {
    "train": ["2017-01-01", "2017-01-10"],
    "validation": ["2017-01-11", "2017-01-15"],
    "test": ["2017-01-16", "2017-01-30"],
}


def make_frame():
    index = pd.date_range("2017-01-01", "2017-01-30 23:50", freq="10min")
    return pd.DataFrame({"x": range(len(index))}, index=index)


def test_blocks_are_in_time_order_and_never_overlap():
    parts = time_split(make_frame(), SPLIT)
    assert parts["train"].index.max() < parts["validation"].index.min()
    assert parts["validation"].index.max() < parts["test"].index.min()


def test_no_rows_lost_or_duplicated():
    df = make_frame()
    parts = time_split(df, SPLIT)
    total = sum(len(p) for p in parts.values())
    assert total == len(df)


def test_end_date_is_included():
    parts = time_split(make_frame(), SPLIT)
    assert parts["train"].index.max() == pd.Timestamp("2017-01-10 23:50")


def test_rows_stay_in_original_order():
    parts = time_split(make_frame(), SPLIT)
    for part in parts.values():
        assert part.index.is_monotonic_increasing


def test_overlapping_ranges_are_rejected():
    bad = {**SPLIT, "validation": ["2017-01-09", "2017-01-15"]}
    with pytest.raises(ValueError):
        time_split(make_frame(), bad)