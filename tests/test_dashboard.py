import json

import numpy as np
import pandas as pd

from src.turbine_ew.dashboard import build_payload, render_html

TARGET = "gen_bearing_rear_temp"


def make_cfg(fault_start="2021-02-05 08:46:16"):
    return {
        "event": {
            "turbine": 1,
            "fault_start": fault_start,
            "split": {
                "validation": ["2020-07-01", "2020-10-31"],
                "test": ["2020-11-01", "2021-02-28"],
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


def make_daily(drift_days=0):
    idx = pd.date_range("2020-07-01", "2021-02-28", freq="D")
    rng = np.random.default_rng(1)
    per_turbine = {t: pd.Series(rng.normal(0, 0.1, len(idx)), index=idx) for t in range(1, 5)}
    if drift_days:
        drift = pd.date_range("2021-01-20", periods=drift_days)
        per_turbine[1].loc[drift] += 5.0
    per_turbine[2].iloc[10] = np.nan
    return {TARGET: per_turbine}


def test_payload_alerts_match_the_scored_alerts():
    payload = build_payload(make_cfg(), make_daily(drift_days=6))
    assert len(payload["alerts"]) == 1
    a = payload["alerts"][0]
    assert a["turbine"] == 1
    assert a["alertDate"] == "2021-01-22"
    assert a["leadDays"] == 14


def test_payload_without_a_fault_has_no_lead_time_or_fault_marker():
    payload = build_payload(make_cfg(fault_start=None), make_daily(drift_days=6))
    assert payload["faultStart"] is None
    assert payload["alerts"][0]["leadDays"] is None


def test_series_line_up_with_the_dates_and_use_null_for_missing_days():
    payload = build_payload(make_cfg(), make_daily())
    assert payload["dates"][0] == "2020-07-01"
    assert payload["dates"][-1] == "2021-02-28"
    for per_turbine in payload["series"][TARGET].values():
        assert len(per_turbine["values"]) == len(payload["dates"])
    assert payload["series"][TARGET]["2"]["values"][10] is None
    assert payload["alerts"] == []


def test_threshold_is_at_least_the_floor():
    payload = build_payload(make_cfg(), make_daily())
    for per_turbine in payload["series"][TARGET].values():
        assert per_turbine["threshold"] >= 2.0


def test_rendered_page_embeds_valid_json_and_no_placeholder():
    payload = build_payload(make_cfg(), make_daily(drift_days=6))
    page = render_html(payload)
    assert "DATAGOESHERE" not in page
    start = page.index('type="application/json">') + len('type="application/json">')
    end = page.index("</script>", start)
    assert json.loads(page[start:end])["alerts"][0]["turbine"] == 1
