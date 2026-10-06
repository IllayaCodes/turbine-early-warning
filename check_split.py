import yaml

from src.turbine_ew.data import load_turbine_years
from src.turbine_ew.preprocess import clean
from src.turbine_ew.split import time_split

with open("configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

d = cfg["data"]
df = load_turbine_years(d["raw_dir"], d["years"], 1, cfg["signals"], d["skip_rows"])
out, report = clean(df, cfg["cleaning"])
parts = time_split(out, cfg["split"])
for name, part in parts.items():
    print(name.ljust(10), len(part), "rows", part.index.min(), "to", part.index.max())
print("rows kept by cleaning:", len(out), " rows across the three blocks:", sum(len(p) for p in parts.values()))