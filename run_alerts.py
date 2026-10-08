import pandas as pd
import yaml

from src.turbine_ew.alerts import fleet_relative
from src.turbine_ew.pipeline import daily_residuals, event_alerts

with open("configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

names = ["linear", "boosting"]
daily = daily_residuals(cfg, names)

test_start, test_end = cfg["event"]["split"]["test"]
months = (pd.Timestamp(test_end) - pd.Timestamp(test_start)).days / 30.44
n_turbines = 6
n_series = n_turbines * len(cfg["model"]["targets"])

summary = []
for name in names:
    thr, alerts = event_alerts(cfg, daily[name])
    alerts.to_csv(f"docs/alerts_{name}.csv", index=False)
    hits = alerts[alerts["lead_days"].notna()]
    other = alerts[alerts["lead_days"].isna()]
    print()
    print("=====", name, "=====")
    print("thresholds (C):")
    print(thr.pivot(index="turbine", columns="target", values="threshold_C").to_string())
    print("alerts:")
    print(alerts.to_string(index=False))
    t5_rear = other[(other["turbine"] == 5) & (other["target"] == "gen_bearing_rear_temp")]
    summary.append({
        "model": name,
        "hits": len(hits),
        "lead_days": hits["lead_days"].max() if len(hits) else None,
        "unmatched": len(other),
        "per_turbine_month": round(len(other) / (months * n_turbines), 2),
        "per_sensor_month": round(len(other) / (months * n_series), 3),
        "turbine5_rear": len(t5_rear),
    })

print()
print("summary over the test block", test_start, "to", test_end, "(about", round(months, 1), "months)")
print(pd.DataFrame(summary).to_string(index=False))

focus = 5
cols = {}
for target in cfg["model"]["targets"]:
    rel = fleet_relative(pd.DataFrame(daily["boosting"][target]), cfg["alerts"]["min_other_turbines"])
    cols[target] = rel[focus]
weekly = pd.DataFrame(cols).resample("W").mean()
print()
print("turbine", focus, "weekly mean residual minus the other turbines (boosting):")
print(weekly.loc["2021-01-10":"2021-05-31"].round(1).to_string())