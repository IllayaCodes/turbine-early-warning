import numpy as np
import pandas as pd

from src.turbine_ew.alerts import merge_episodes
from src.turbine_ew.pipeline import event_alerts


def make_alerts(rows):
    df = pd.DataFrame(rows, columns=["turbine", "target", "start", "end", "lead_days"])
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    return df


def test_close_alerts_merge_into_one_episode():
    alerts = make_alerts([
        (5, "rear", "2021-01-01", "2021-01-03", np.nan),
        (5, "rear", "2021-01-08", "2021-01-10", np.nan),
    ])
    out = merge_episodes(alerts, max_gap_days=7)
    assert len(out) == 1
    assert out.iloc[0]["n_alerts"] == 2
    assert out.iloc[0]["end"] == pd.Timestamp("2021-01-10")


def test_distant_alerts_stay_separate():
    alerts = make_alerts([
        (5, "rear", "2021-01-01", "2021-01-03", np.nan),
        (5, "rear", "2021-01-15", "2021-01-17", np.nan),
    ])
    assert len(merge_episodes(alerts, max_gap_days=7)) == 2


def test_different_turbines_or_targets_never_merge():
    alerts = make_alerts([
        (1, "rear", "2021-01-01", "2021-01-03", np.nan),
        (2, "rear", "2021-01-04", "2021-01-06", np.nan),
        (1, "front", "2021-01-04", "2021-01-06", np.nan),
    ])
    assert len(merge_episodes(alerts, max_gap_days=7)) == 3


def test_a_lead_time_is_kept_when_merging():
    alerts = make_alerts([
        (6, "rear", "2021-01-01", "2021-01-03", np.nan),
        (6, "rear", "2021-01-05", "2021-01-07", 15.0),
    ])
    out = merge_episodes(alerts, max_gap_days=7)
    assert len(out) == 1
    assert out.iloc[0]["lead_days"] == 15.0


def test_no_alerts_gives_an_empty_table():
    out = merge_episodes(make_alerts([]), max_gap_days=7)
    assert len(out) == 0


def test_a_section_without_a_fault_gives_no_lead_times():
    idx = pd.date_range("2018-07-01", "2019-03-31", freq="D")
    rng = np.random.default_rng(0)
    data = {t: pd.Series(rng.normal(0, 0.1, len(idx)), index=idx) for t in range(1, 5)}
    data[1].loc[pd.date_range("2019-02-01", "2019-02-05")] += 5.0
    cfg = {
        "period2": {
            "turbine": 1,
            "fault_start": None,
            "split": {
                "validation": ["2018-07-01", "2018-09-30"],
                "test": ["2018-11-01", "2019-03-31"],
            },
        },
        "alerts": {
            "min_other_turbines": 3,
            "k": 4.0,
            "floor_c": 2.0,
            "min_validation_days": 30,
            "consecutive_days": 3,
            "window_days": 30,
        },
    }
    _, alerts = event_alerts(cfg, {"gen_bearing_rear_temp": data}, section="period2")
    assert len(alerts) == 1
    assert pd.isna(alerts.iloc[0]["lead_days"])