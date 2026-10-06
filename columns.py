import sys
import pandas as pd
from pathlib import Path

path = Path(sys.argv[1])
df = pd.read_csv(path, skiprows=9, na_values=["NaN"], low_memory=False)
df = df.rename(columns={df.columns[0]: "time"})
df["time"] = pd.to_datetime(df["time"])
Path("docs/columns_clean.txt").write_text("\n".join(df.columns), encoding="utf-8")

print(len(df), "rows,", len(df.columns), "columns")
print("from", df["time"].min(), "to", df["time"].max())

skip = (", Standard deviation", ", Minimum", ", Maximum")
base = [c for c in df.columns if not any(s in c for s in skip)]
print()
print(len(base), "average columns (the main signals):")
for i, c in enumerate(base):
    print(i, c)

prefixes = ["Wind speed (", "Power (", "Rotor speed (", "Gearbox speed (",
            "Nacelle ambient temperature (", "Ambient temperature (converter)",
            "Gear oil temperature (", "Gear oil inlet temperature (",
            "Generator bearing front temperature (", "Generator bearing rear temperature (",
            "Front bearing temperature (", "Rear bearing temperature (",
            "Stator temperature 1 (", "Rotor bearing temp ("]
print()
print("share of rows with a value: overall, then Jan..Dec")
for p in prefixes:
    for c in df.columns:
        if c.startswith(p):
            s = df[c].notna()
            bym = s.groupby(df["time"].dt.month).mean().mul(100).round(0).astype(int).tolist()
            print(c[:42].ljust(42), str(round(s.mean() * 100, 1)).rjust(5) + "%", bym)
