import pandas as pd

from src.turbine_ew.preprocess import clean

CLEANING = {
    "min_wind_speed": 3.0,
    "max_wind_speed": 25.0,
    "min_power": 0.0,
    "min_rotor_speed": 5.0,
    "limits": {"rotor_speed": [0, 20], "gear_oil_temp": [0, 90]},
}

HEALTHY = {
    "wind_speed": 8.0,
    "power": 900.0,
    "rotor_speed": 13.0,
    "gearbox_speed": 1300.0,
    "ambient_temp": 10.0,
    "gear_oil_temp": 55.0,
}


def make_frame(overrides):
    """One row per dict in overrides, each starting from a healthy row."""
    data = [{**HEALTHY, **o} for o in overrides]
    index = pd.date_range("2017-01-01", periods=len(data), freq="10min")
    return pd.DataFrame(data, index=index)


def test_healthy_row_is_kept_and_time_stays_ordered():
    out, report = clean(make_frame([{}, {}, {}]), CLEANING)
    assert report["rows_out"] == 3
    assert out.index.is_monotonic_increasing


def test_stopped_turbine_is_dropped():
    out, report = clean(make_frame([{"power": 0.0, "rotor_speed": 0.0}]), CLEANING)
    assert len(out) == 0
    assert report["not_operating"] == 1


def test_wind_below_cut_in_is_dropped():
    out, report = clean(make_frame([{"wind_speed": 2.0}]), CLEANING)
    assert len(out) == 0
    assert report["not_operating"] == 1


def test_impossible_rotor_speed_is_dropped():
    out, report = clean(make_frame([{"rotor_speed": 49.0}]), CLEANING)
    assert len(out) == 0
    assert report["implausible"] == 1


def test_missing_values_are_dropped_not_filled():
    out, report = clean(make_frame([{"gear_oil_temp": float("nan")}]), CLEANING)
    assert len(out) == 0
    assert report["missing"] == 1
    assert out.isna().sum().sum() == 0


def test_report_counts_add_up():
    rows = [{}, {"power": 0.0}, {"rotor_speed": 49.0}, {"gear_oil_temp": float("nan")}, {"wind_speed": 2.0}]
    out, report = clean(make_frame(rows), CLEANING)
    assert report["rows_in"] == 5
    assert report["rows_out"] == 1
    assert report["missing"] + report["implausible"] + report["not_operating"] + report["rows_out"] == 5