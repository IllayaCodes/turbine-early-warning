import sys
from pathlib import Path

import pandas as pd
import yaml

KEYWORDS = "temp|heat|fan|cool"

with open("configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

csv_path = sys.argv[1]
section = sys.argv[2]
alerts = pd.read_csv(csv_path, parse_dates=["start", "alert_date", "end"])
day = pd.Timedelta(days=1)

frames = []
for year in cfg[section]["years"]:
    folder = Path(cfg["data"]["raw_dir"]) / f"kelmarsh{year}"
    for f in sorted(folder.glob("Status_Kelmarsh_*.csv")):
        d = pd.read_csv(f, skiprows=9)
        d["turbine"] = int(f.name.split("_")[2])
        frames.append(d)
log = pd.concat(frames, ignore_index=True)
log["s"] = pd.to_datetime(log["Timestamp start"], errors="coerce")
is_thermal = log["Status"].isin(["Warning", "Stop"]) & log["Message"].str.contains(KEYWORDS, case=False, na=False)
thermal = log[is_thermal]

rows = []
for _, a in alerts.iterrows():
    mine = thermal[
        (thermal["turbine"] == a["turbine"])
        & (thermal["s"] >= a["start"] - day)
        & (thermal["s"] < a["end"] + 2 * day)
    ]
    if len(mine) == 0:
        continue
    first = mine.sort_values("s").iloc[0]
    rows.append({
        "turbine": a["turbine"],
        "target": a["target"].replace("gen_bearing_", "").replace("_temp", ""),
        "alert_date": a["alert_date"].date(),
        "first_message": first["s"],
        "message": first["Message"],
        "alert_days_before_message": (first["s"].normalize() - a["alert_date"]).days,
    })

if rows:
    print(pd.DataFrame(rows).to_string(index=False))
else:
    print("no alerts with a thermal message nearby")