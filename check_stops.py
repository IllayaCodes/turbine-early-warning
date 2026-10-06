import sys
from pathlib import Path

import pandas as pd

folder = Path(sys.argv[1])
files = sorted(folder.rglob("Status*.csv"))
if not files:
    raise SystemExit(f"no Status*.csv files found under {folder} - check the folder name")

frames = []
for f in files:
    d = pd.read_csv(f, skiprows=9)
    d["turbine"] = int(f.name.split("_")[2])
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
df["start"] = pd.to_datetime(df["Timestamp start"], errors="coerce")
df["end"] = pd.to_datetime(df["Timestamp end"], errors="coerce")
df["hours"] = ((df["end"] - df["start"]).dt.total_seconds() / 3600).round(0)
stops = df[(df["Status"] == "Stop") & (df["hours"] > 24)]
cols = ["turbine", "start", "end", "hours", "Code", "Message"]
print(stops.sort_values(["turbine", "start"])[cols].to_string(index=False))