import pandas as pd
import yaml

from src.turbine_ew.alerts import merge_episodes
from src.turbine_ew.pipeline import daily_residuals, event_alerts

SECTION = "period2"
MAX_GAP_DAYS = 7

with open("configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

names = ["linear", "boosting"]
daily = daily_residuals(cfg, names, section=SECTION)

test_start, test_end = cfg[SECTION]["split"]["test"]
months = (pd.Timestamp(test_end) - pd.Timestamp(test_start)).days / 30.44
n_series = 6 * len(cfg["model"]["targets"])

summary = []
for name in names:
    thr, alerts = event_alerts(cfg, daily[name], section=SECTION)
    alerts.to_csv(f"docs/alerts_{SECTION}_{name}.csv", index=False)
    stamped = alerts.assign(start=pd.to_datetime(alerts["start"]), end=pd.to_datetime(alerts["end"]))
    episodes = merge_episodes(stamped, MAX_GAP_DAYS)
    print()
    print("=====", name, "=====")
    print("thresholds (C), floor is 2.0:")
    print(thr.pivot(index="turbine", columns="target", values="threshold_C").to_string())
    print("alerts:")
    print(alerts.to_string(index=False))
    if len(episodes) > 0:
        shown = episodes.copy()
        shown["days"] = (shown["end"] - shown["start"]).dt.days + 1
        print("episodes per turbine and sensor:")
        print(shown.groupby(["turbine", "target"]).size().to_string())
        print("episode starts per month:")
        print(shown["start"].dt.to_period("M").value_counts().sort_index().to_string())
    summary.append({
        "model": name,
        "alerts": len(alerts),
        "episodes": len(episodes),
        "per_turbine_month": round(len(episodes) / (months * 6), 2),
        "per_sensor_month": round(len(episodes) / (months * n_series), 3),
    })

print()
print("period 2 test block", test_start, "to", test_end, "(about", round(months, 1), "months), no labelled fault")
print(pd.DataFrame(summary).to_string(index=False))