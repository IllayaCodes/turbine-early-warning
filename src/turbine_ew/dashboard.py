import json
from pathlib import Path

import pandas as pd

from src.turbine_ew.alerts import fleet_relative
from src.turbine_ew.pipeline import event_alerts

TEMPLATE = Path("src") / "turbine_ew" / "dashboard.html"
PLACEHOLDER = "DATAGOESHERE"


def format_number(value):
    return None if pd.isna(value) else round(float(value), 2)


def short_label(target):
    return target.replace("gen_", "").replace("_temp", "").replace("_", " ")


def build_payload(cfg, daily_by_target, section="event", model_name="linear"):
    """Everything the dashboard shows, as plain numbers and strings.

    Thresholds and alerts come from event_alerts, so the dashboard can never
    disagree with the scored results. The series are the fleet-relative daily
    residuals from the start of validation to the end of the test block.
    """
    ev = cfg[section]
    al = cfg["alerts"]
    first = ev["split"]["validation"][0]
    last = ev["split"]["test"][1]
    thresholds, alerts = event_alerts(cfg, daily_by_target, section)
    limit = {}
    for _, row in thresholds.iterrows():
        limit[(row["turbine"], row["target"])] = float(row["threshold_C"])

    dates = None
    series = {}
    for target, per_turbine in daily_by_target.items():
        rel = fleet_relative(pd.DataFrame(per_turbine), al["min_other_turbines"]).loc[first:last]
        dates = [d.strftime("%Y-%m-%d") for d in rel.index]
        series[target] = {}
        for turbine in rel.columns:
            series[target][str(turbine)] = {
                "threshold": limit.get((turbine, target)),
                "values": [format_number(v) for v in rel[turbine]],
            }

    turbines = sorted({int(t) for per_turbine in daily_by_target.values() for t in per_turbine})
    fault = ev.get("fault_start")
    records = []
    for _, a in alerts.iterrows():
        records.append({
            "turbine": int(a["turbine"]),
            "target": a["target"],
            "start": str(a["start"]),
            "alertDate": str(a["alert_date"]),
            "end": str(a["end"]),
            "leadDays": None if pd.isna(a["lead_days"]) else int(a["lead_days"]),
        })
    return {
        "modelName": model_name,
        "section": section,
        "dates": dates or [],
        "validation": list(ev["split"]["validation"]),
        "test": list(ev["split"]["test"]),
        "faultStart": str(fault) if fault else None,
        "faultTurbine": int(ev["turbine"]) if fault else None,
        "rule": {
            "k": al["k"],
            "floorC": al["floor_c"],
            "consecutiveDays": al["consecutive_days"],
            "windowDays": al["window_days"],
        },
        "turbines": turbines,
        "targets": list(daily_by_target),
        "labels": {t: short_label(t) for t in daily_by_target},
        "series": series,
        "alerts": records,
    }


def render_html(payload):
    """The finished page: one self-contained HTML file, no internet needed to open it."""
    text = TEMPLATE.read_text(encoding="utf-8")
    data = json.dumps(payload).replace("</", "<\\/")
    return text.replace(PLACEHOLDER, data)


def write_dashboard(payload, path):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(payload), encoding="utf-8")
    return out
