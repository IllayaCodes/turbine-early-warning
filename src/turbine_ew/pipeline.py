import warnings

import pandas as pd

from src.turbine_ew.alerts import daily_mean, find_alerts, fleet_relative, lead_days, threshold
from src.turbine_ew.data import load_turbine_years
from src.turbine_ew.models import fit_model, make_boosting, make_linear, residuals
from src.turbine_ew.preprocess import clean
from src.turbine_ew.split import time_split

MAKERS = {"linear": make_linear, "boosting": make_boosting}
ALERT_COLUMNS = ["turbine", "target", "start", "alert_date", "end", "days", "lead_days"]


def daily_residuals(cfg, model_names, turbines=range(1, 7), section="event"):
    """Daily mean residual per model, target and turbine, for the section's years.

    Each model is trained on the section's train block only.
    Returns {model_name: {target: {turbine: daily Series}}}.
    """
    d = cfg["data"]
    ev = cfg[section]
    min_rows = cfg["alerts"]["min_rows_per_day"]
    features = cfg["model"]["features"]
    targets = cfg["model"]["targets"]
    daily = {m: {t: {} for t in targets} for m in model_names}
    for turbine in turbines:
        df = load_turbine_years(d["raw_dir"], ev["years"], turbine, cfg["signals"], d["skip_rows"])
        out, _ = clean(df, cfg["cleaning"])
        parts = time_split(out, ev["split"])
        for name in model_names:
            for target in targets:
                model = fit_model(MAKERS[name](), parts["train"], features, target)
                res = residuals(model, out, features, target)
                daily[name][target][turbine] = daily_mean(res, min_rows)
        print("done turbine", turbine, flush=True)
    return daily


def event_alerts(cfg, daily_by_target, section="event"):
    """Thresholds (from validation days only) and alerts (in the test block) for one model."""
    ev = cfg[section]
    al = cfg["alerts"]
    fault = ev.get("fault_start")
    val_start, val_end = ev["split"]["validation"]
    test_start, test_end = ev["split"]["test"]
    thresholds = []
    rows = []
    for target, per_turbine in daily_by_target.items():
        rel = fleet_relative(pd.DataFrame(per_turbine), al["min_other_turbines"])
        for turbine in rel.columns:
            series = rel[turbine]
            try:
                thr = threshold(series.loc[val_start:val_end], al["k"], al["floor_c"], al["min_validation_days"])
            except ValueError as err:
                warnings.warn(f"no threshold for turbine {turbine}, {target}: {err}")
                continue
            thresholds.append({"turbine": turbine, "target": target, "threshold_C": round(thr, 2)})
            for a in find_alerts(series.loc[test_start:test_end], thr, al["consecutive_days"]):
                lead = None
                if fault and turbine == ev["turbine"]:
                    lead = lead_days(a["alert_date"], fault, al["window_days"])
                rows.append({
                    "turbine": turbine,
                    "target": target,
                    "start": a["start"].date(),
                    "alert_date": a["alert_date"].date(),
                    "end": a["end"].date(),
                    "days": (a["end"] - a["start"]).days + 1,
                    "lead_days": lead,
                })
    return pd.DataFrame(thresholds), pd.DataFrame(rows, columns=ALERT_COLUMNS)