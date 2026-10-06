import pandas as pd
import yaml

from src.turbine_ew.data import load_turbine_years
from src.turbine_ew.models import fit_model, make_boosting, residuals
from src.turbine_ew.preprocess import clean
from src.turbine_ew.split import time_split

with open("configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

d = cfg["data"]
df = load_turbine_years(d["raw_dir"], d["years"], 1, cfg["signals"], d["skip_rows"])
out, _ = clean(df, cfg["cleaning"])
parts = time_split(out, cfg["split"])

features = cfg["model"]["features"]
monthly = {}
for target in cfg["model"]["targets"]:
    model = fit_model(make_boosting(), parts["train"], features, target)
    res = residuals(model, out, features, target)
    monthly[target] = res.groupby([out.index.year, out.index.month]).mean()
table = pd.DataFrame(monthly)
table.index.names = ["year", "month"]
table["rows"] = out.groupby([out.index.year, out.index.month]).size()
print(table.round(2).to_string())