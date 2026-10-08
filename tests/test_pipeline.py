import numpy as np
import pandas as pd

from src.turbine_ew.pipeline import event_alerts

CFG = {
    "event": {
        "turbine": 1,
        "fault_start": "2021-02-20 00:00:00",
        "split": {
            "validation": ["2020-07-01", "2020-09-30"],
            "test": ["2020-11-01", "2021-03-31"],
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


def make_daily(bump_days=None):
    idx = pd.date_range("2020-07-01", "2021-03-31", freq="D")
    rng = np.random.default_rng(0)
    data = {t: pd.Series(rng.normal(0, 0.1, len(idx)), index=idx) for t in range(1, 5)}
    if bump_days is not None:
        data[1].loc[bump_days] += 5.0
    return {"gen_bearing_rear_temp": data}


def test_a_bump_gives_one_alert_with_a_lead_time():
    daily = make_daily(pd.date_range("2021-02-01", "2021-02-05"))
    thresholds, alerts = event_alerts(CFG, daily)
    assert len(thresholds) == 4
    assert len(alerts) == 1
    row = alerts.iloc[0]
    assert row["turbine"] == 1
    assert row["lead_days"] == 17


def test_no_bump_gives_no_alerts():
    _, alerts = event_alerts(CFG, make_daily())
    assert len(alerts) == 0


def test_an_alert_far_from_the_fault_has_no_lead_time():
    event = dict(CFG["event"], fault_start="2021-03-30 00:00:00")
    cfg = dict(CFG, event=event)
    _, alerts = event_alerts(cfg, make_daily(pd.date_range("2021-02-01", "2021-02-05")))
    assert len(alerts) == 1
    assert pd.isna(alerts.iloc[0]["lead_days"])