import pandas as pd
import yaml

from src.turbine_ew.data import load_turbine_years
from src.turbine_ew.models import error_summary, fit_model, make_linear, residuals
from src.turbine_ew.preprocess import clean
from src.turbine_ew.split import time_split

with open("configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

d = cfg["data"]
df = load_turbine_years(d["raw_dir"], d["years"], 1, cfg["signals"], d["skip_rows"])
out, _ = clean(df, cfg["cleaning"])
parts = time_split(out, cfg["split"])

features = cfg["model"]["features"]
rows = []
for target in cfg["model"]["targets"]:
    model = fit_model(make_linear(), parts["train"], features, target)
    for name, part in parts.items():
        s = error_summary(residuals(model, part, features, target))
        rows.append({"target": target, "block": name, **s})
print(pd.DataFrame(rows).round(2).to_string(index=False))