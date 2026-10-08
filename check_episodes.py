import pandas as pd
import yaml

MAX_GAP_DAYS = 7

with open("configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

test_start, test_end = cfg["event"]["split"]["test"]
months = (pd.Timestamp(test_end) - pd.Timestamp(test_start)).days / 30.44

summary = []
for name in ["linear", "boosting"]:
    alerts = pd.read_csv(f"docs/alerts_{name}.csv", parse_dates=["start", "end"])
    alerts = alerts.sort_values(["turbine", "target", "start"])
    rows = []
    for (turbine, target), group in alerts.groupby(["turbine", "target"]):
        cur = None
        for _, a in group.iterrows():
            if cur is not None and (a["start"] - cur["end"]).days <= MAX_GAP_DAYS:
                cur["end"] = max(cur["end"], a["end"])
                cur["n_alerts"] += 1
                if pd.isna(cur["lead_days"]):
                    cur["lead_days"] = a["lead_days"]
            else:
                if cur is not None:
                    rows.append(cur)
                cur = {
                    "turbine": turbine,
                    "target": target,
                    "start": a["start"],
                    "end": a["end"],
                    "n_alerts": 1,
                    "lead_days": a["lead_days"],
                }
        rows.append(cur)
    episodes = pd.DataFrame(rows)
    episodes["days"] = (episodes["end"] - episodes["start"]).dt.days + 1
    episodes["start"] = episodes["start"].dt.date
    episodes["end"] = episodes["end"].dt.date
    hits = int(episodes["lead_days"].notna().sum())
    unmatched = len(episodes) - hits
    summary.append({
        "model": name,
        "alerts": len(alerts),
        "episodes": len(episodes),
        "hits": hits,
        "unmatched_episodes": unmatched,
        "per_turbine_month": round(unmatched / (months * 6), 2),
    })
    print()
    print("=====", name, "=====")
    print(episodes[["turbine", "target", "start", "end", "days", "n_alerts", "lead_days"]].to_string(index=False))

print()
print("merging alerts with a gap of at most", MAX_GAP_DAYS, "days; test block", test_start, "to", test_end)
print(pd.DataFrame(summary).to_string(index=False))