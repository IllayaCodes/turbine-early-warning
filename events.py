import sys
import pandas as pd
from pathlib import Path

folder = Path(sys.argv[1])
frames = []
for f in sorted(folder.glob("Status*.csv")):
    d = pd.read_csv(f, skiprows=9)
    d["turbine"] = f.name.split("_")[2]
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
df["start"] = pd.to_datetime(df["Timestamp start"], errors="coerce")
df["end"] = pd.to_datetime(df["Timestamp end"], errors="coerce")
df["hours"] = (df["end"] - df["start"]).dt.total_seconds() / 3600

print("all events:", len(df))
print("events with no usable end time:", int(df["end"].isna().sum()))
print("events with no usable start time:", int(df["start"].isna().sum()))
print(df["Status"].value_counts().to_string())

stops = df[df["Status"] == "Stop"]
print()
print("Stop events by code and message (count, total hours):")
g = stops.groupby(["Code", "Message"]).agg(n=("hours", "size"), hours=("hours", "sum"))
print(g.sort_values("hours", ascending=False).round(1).to_string())

kw = "gear|generator|bearing|temp|overheat|vibration|converter"
rel = stops[stops["Message"].str.contains(kw, case=False, na=False)]
print()
print("Stop events mentioning gear/generator/bearing/oil/temp:")
cols = ["turbine", "start", "end", "hours", "Code", "Message", "IEC category"]
print(rel.assign(hours=rel["hours"].round(1))[cols].to_string())