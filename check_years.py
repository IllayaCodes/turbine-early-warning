import yaml

from src.turbine_ew.data import load_turbine_years
from src.turbine_ew.preprocess import clean

with open("configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

d = cfg["data"]
df = load_turbine_years(d["raw_dir"], d["years"], 1, cfg["signals"], d["skip_rows"])
print("rows:", len(df), "from", df.index.min(), "to", df.index.max())
print("duplicate timestamps:", int(df.index.duplicated().sum()))

out, report = clean(df, cfg["cleaning"])
print(report)

kept = out.groupby([out.index.year, out.index.month]).size().unstack(fill_value=0)
print("rows kept per year (rows) and month:")
print(kept)