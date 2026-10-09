import json
import sys

import yaml

from src.turbine_ew.data import load_turbine_years
from src.turbine_ew.preprocess import clean

when = sys.argv[1] if len(sys.argv) > 1 else "2021-01-19 12:00"
turbine = int(sys.argv[2]) if len(sys.argv) > 2 else 6

with open("configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

d = cfg["data"]
df = load_turbine_years(d["raw_dir"], [2021], turbine, cfg["signals"], d["skip_rows"])
out, _ = clean(df, cfg["cleaning"])
row = out.loc[when:].iloc[0]
body = {
    "turbine": turbine,
    "wind_speed": round(float(row["wind_speed"]), 2),
    "power": round(float(row["power"]), 1),
    "rotor_speed": round(float(row["rotor_speed"]), 2),
    "gearbox_speed": round(float(row["gearbox_speed"]), 1),
    "ambient_temp": round(float(row["ambient_temp"]), 1),
    "actual": {t: round(float(row[t]), 1) for t in cfg["model"]["targets"]},
}
print("row time:", row.name)
print(json.dumps(body))