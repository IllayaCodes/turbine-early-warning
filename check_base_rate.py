import sys
from pathlib import Path

import pandas as pd
import yaml

KEYWORDS = "temp|heat|fan|cool"

with open("configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

csv_path = sys.argv[1]
section = sys.argv[2]
alerts = pd.read_csv(csv_path, parse_dates=["start", "end"])
test_start = pd.Timestamp(cfg[section]["split"]["test"][0])
test_end = pd.Timestamp(cfg[section]["split"]["test"][1])
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

index = pd.date_range(test_start - day, test_end + day, freq="D")
flags = {}
for turbine in sorted(alerts["turbine"].unique()):
    days = thermal[thermal["turbine"] == turbine]["s"].dt.normalize()
    flags[turbine] = pd.Series(index.isin(days).astype(int), index=index)

rows = []
for _, a in alerts.iterrows():
    flag = flags[a["turbine"]]
    length = int(a["days"]) + 2
    chance = float(flag.rolling(length).max().dropna().mean())
    seen = int(flag.loc[a["start"] - day : a["end"] + day].max())
    rows.append({
        "turbine": a["turbine"],
        "target": a["target"].replace("gen_bearing_", "").replace("_temp", ""),
        "start": a["start"].date(),
        "window_days": length,
        "thermal_nearby": seen,
        "chance": round(chance, 2),
    })

out = pd.DataFrame(rows)
print(out.to_string(index=False))
print()
print("alerts:", len(out))
print("alerts with a thermal message nearby:", int(out["thermal_nearby"].sum()))
print("expected by chance (sum of the per-alert chances):", round(float(out["chance"].sum()), 1))