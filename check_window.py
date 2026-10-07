import sys
from pathlib import Path

import pandas as pd

folder = Path(sys.argv[1])
turbine = int(sys.argv[2])
start = pd.Timestamp(sys.argv[3])
end = pd.Timestamp(sys.argv[4])

files = sorted(folder.rglob(f"Status_Kelmarsh_{turbine}_*.csv"))
if not files:
    raise SystemExit(f"no status file for turbine {turbine} under {folder}")

df = pd.read_csv(files[0], skiprows=9)
df["start"] = pd.to_datetime(df["Timestamp start"], errors="coerce")
df["end"] = pd.to_datetime(df["Timestamp end"], errors="coerce")
df = df[(df["start"] >= start) & (df["start"] <= end)]
df = df[(df["Status"] != "Informational") | df["Comment"].notna()]
cols = ["start", "end", "Status", "Code", "Message", "Comment", "Service contract category"]
print(len(df), "events (not Informational, or with a comment)")
print(df[cols].head(60).to_string(index=False))