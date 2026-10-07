import numpy as np
import pandas as pd
import pytest

from src.turbine_ew.alerts import (
    daily_mean,
    find_alerts,
    fleet_relative,
    lead_days,
    threshold,
)


def make_rel(values):
    idx = pd.date_range("2021-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


def test_daily_mean_masks_days_with_too_few_rows():
    idx = list(pd.date_range("2021-01-01", periods=24, freq="60min"))
    idx += list(pd.date_range("2021-01-02", periods=5, freq="60min"))
    out = daily_mean(pd.Series(2.0, index=idx), min_rows=10)
    assert out.loc["2021-01-01"] == 2.0
    assert np.isnan(out.loc["2021-01-02"])


def test_fleet_relative_subtracts_the_median_of_the_others():
    days = pd.date_range("2021-01-01", periods=2, freq="D")
    daily = pd.DataFrame({1: [5.0, 1.0], 2: [1.0, 1.0], 3: [2.0, 1.0], 4: [3.0, 1.0]}, index=days)
    rel = fleet_relative(daily, min_others=3)
    assert rel.loc[days[0], 1] == 3.0
    assert rel.loc[days[1], 1] == 0.0


def test_fleet_relative_needs_enough_other_turbines():
    days = pd.date_range("2021-01-01", periods=1, freq="D")
    daily = pd.DataFrame({1: [5.0], 2: [1.0], 3: [np.nan], 4: [np.nan]}, index=days)
    rel = fleet_relative(daily, min_others=2)
    assert np.isnan(rel.loc[days[0], 1])


def test_threshold_uses_the_floor_when_the_spread_is_tiny():
    x = pd.Series([0.0, 0.01, -0.01, 0.0] * 10)
    assert threshold(x, k=4, floor=2.0, min_days=10) == 2.0


def test_threshold_scales_with_spread():
    x = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0] * 10)
    assert threshold(x, k=4, floor=0.5, min_days=10) == pytest.approx(4 * 1.4826)


def test_threshold_refuses_too_little_data():
    with pytest.raises(ValueError):
        threshold(pd.Series([0.1, 0.2]), k=4, floor=2.0, min_days=10)


def test_run_of_three_gives_an_alert_on_the_third_day():
    alerts = find_alerts(make_rel([0, 5, 5, 5, 5, 0]), thr=2.0, n_days=3)
    assert len(alerts) == 1
    assert alerts[0]["start"] == pd.Timestamp("2021-01-02")
    assert alerts[0]["alert_date"] == pd.Timestamp("2021-01-04")
    assert alerts[0]["end"] == pd.Timestamp("2021-01-05")


def test_a_run_shorter_than_n_days_gives_no_alert():
    assert find_alerts(make_rel([0, 5, 5, 0, 5, 5, 0]), thr=2.0, n_days=3) == []


def test_a_missing_day_breaks_the_run():
    assert find_alerts(make_rel([5, 5, np.nan, 5, 5]), thr=2.0, n_days=3) == []


def test_days_must_be_consecutive_calendar_days():
    idx = pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-04"])
    with pytest.raises(ValueError):
        find_alerts(pd.Series([5.0, 5.0, 5.0], index=idx), thr=2.0, n_days=3)


def test_two_separate_runs_give_two_alerts():
    assert len(find_alerts(make_rel([5, 5, 5, 0, 5, 5, 5, 5]), thr=2.0, n_days=3)) == 2


def test_lead_days_inside_the_window():
    assert lead_days("2021-01-21", "2021-02-05 08:46:16", window_days=30) == 15


def test_lead_days_outside_the_window():
    assert lead_days("2020-12-01", "2021-02-05 08:46:16", window_days=30) is None
    assert lead_days("2021-02-06", "2021-02-05 08:46:16", window_days=30) is None