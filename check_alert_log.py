from pathlib import Path

import pandas as pd
import yaml

KEYWORDS = "temp|heat|fan|cool"

with open("configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

alerts = pd.read_csv("docs/alerts_test.csv", parse_dates=["start", "alert_date", "end"])

frames = []
for year in cfg["event"]["years"]:
    folder = Path(cfg["data"]["raw_dir"]) / f"kelmarsh{year}"
    for f in sorted(folder.glob("Status_Kelmarsh_*.csv")):
        d = pd.read_csv(f, skiprows=9)
        d["turbine"] = int(f.name.split("_")[2])
        frames.append(d)
log = pd.concat(frames, ignore_index=True)
log["s"] = pd.to_datetime(log["Timestamp start"], errors="coerce")
log["e"] = pd.to_datetime(log["Timestamp end"], errors="coerce")
log["hours"] = (log["e"] - log["s"]).dt.total_seconds() / 3600

day = pd.Timedelta(days=1)
rows = []
for _, a in alerts.iterrows():
    mine = log[log["turbine"] == a["turbine"]]
    stops = mine[(mine["Status"] == "Stop") & (mine["hours"] >= 12)]
    after_stop = bool(((stops["e"] >= a["start"] - 3 * day) & (stops["e"] < a["start"] + day)).any())
    warn = mine[
        mine["Status"].isin(["Warning", "Stop"])
        & mine["Message"].str.contains(KEYWORDS, case=False, na=False)
        & (mine["s"] >= a["start"] - day)
        & (mine["s"] < a["end"] + 2 * day)
    ]
    others = alerts[(alerts["turbine"] != a["turbine"]) & ((alerts["start"] - a["start"]).abs() <= 3 * day)]
    if pd.notna(a["lead_days"]):
        category = "hit"
    elif after_stop:
        category = "after a long stop"
    elif len(warn) > 0:
        category = "thermal warning in log"
    else:
        category = "unexplained"
    rows.append({
        "turbine": a["turbine"],
        "target": a["target"].replace("gen_bearing_", "").replace("_temp", ""),
        "start": a["start"].date(),
        "days": a["days"],
        "other_alerts_3d": len(others),
        "thermal_msgs": len(warn),
        "top_msgs": "; ".join(warn["Message"].value_counts().head(2).index),
        "category": category,
    })

out = pd.DataFrame(rows)
print(out.to_string(index=False))
print()
print(out["category"].value_counts().to_string())