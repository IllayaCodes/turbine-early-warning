import pandas as pd
import yaml

from src.turbine_ew.alerts import daily_mean, find_alerts, fleet_relative, lead_days, threshold
from src.turbine_ew.data import load_turbine_years
from src.turbine_ew.models import fit_model, make_boosting, residuals
from src.turbine_ew.preprocess import clean
from src.turbine_ew.split import time_split

with open("configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

d = cfg["data"]
ev = cfg["event"]
al = cfg["alerts"]
features = cfg["model"]["features"]
targets = cfg["model"]["targets"]
turbines = list(range(1, 7))

daily = {target: {} for target in targets}
for turbine in turbines:
    df = load_turbine_years(d["raw_dir"], ev["years"], turbine, cfg["signals"], d["skip_rows"])
    out, _ = clean(df, cfg["cleaning"])
    parts = time_split(out, ev["split"])
    for target in targets:
        model = fit_model(make_boosting(), parts["train"], features, target)
        res = residuals(model, out, features, target)
        daily[target][turbine] = daily_mean(res, al["min_rows_per_day"])
    print("done turbine", turbine, flush=True)

val_start, val_end = ev["split"]["validation"]
test_start, test_end = ev["split"]["test"]
thresholds = []
rows = []
for target in targets:
    rel = fleet_relative(pd.DataFrame(daily[target]), al["min_other_turbines"])
    for turbine in turbines:
        series = rel[turbine]
        try:
            thr = threshold(series.loc[val_start:val_end], al["k"], al["floor_c"], al["min_validation_days"])
        except ValueError as err:
            print("no threshold for turbine", turbine, target, "-", err)
            continue
        thresholds.append({"turbine": turbine, "target": target, "threshold_C": round(thr, 2)})
        for a in find_alerts(series.loc[test_start:test_end], thr, al["consecutive_days"]):
            lead = None
            if turbine == ev["turbine"]:
                lead = lead_days(a["alert_date"], ev["fault_start"], al["window_days"])
            rows.append({
                "turbine": turbine,
                "target": target,
                "start": a["start"].date(),
                "alert_date": a["alert_date"].date(),
                "end": a["end"].date(),
                "days": (a["end"] - a["start"]).days + 1,
                "lead_days": lead,
            })

print()
print("thresholds (degrees C, from the validation block):")
print(pd.DataFrame(thresholds).pivot(index="turbine", columns="target", values="threshold_C").to_string())
print()
print("alerts in the test block", test_start, "to", test_end)
if rows:
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    print()
    print("alert count per target:")
    print(table.groupby("target").size().to_string())
else:
    print("no alerts")