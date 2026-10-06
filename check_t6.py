import pandas as pd
import yaml

from src.turbine_ew.data import load_turbine_years
from src.turbine_ew.models import fit_model, make_boosting, residuals
from src.turbine_ew.preprocess import clean
from src.turbine_ew.split import time_split

with open("configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

d = cfg["data"]
signals = {
    **cfg["signals"],
    "converter_air_temp": "Ambient temperature (converter)",
    "nacelle_temp": "Nacelle temperature",
}
targets = cfg["model"]["targets"] + ["converter_air_temp", "nacelle_temp"]
features = cfg["model"]["features"]

df = load_turbine_years(d["raw_dir"], d["years"], 6, signals, d["skip_rows"])
out, _ = clean(df, cfg["cleaning"])
parts = time_split(out, cfg["split"])

weekly = {}
for target in targets:
    model = fit_model(make_boosting(), parts["train"], features, target)
    res = residuals(model, out, features, target)
    weekly[target] = res.resample("W").mean()
table = pd.DataFrame(weekly)

print("rough normal weekly spread (std of weekly means, validation block Aug-Oct 2017):")
print(table.loc["2017-08-01":"2017-10-31"].std().round(2).to_string())
print()
print("turbine 6 weekly mean residual (converter error: 2017-11-11 to 2017-11-16):")
print(table.loc["2017-09-03":"2017-12-24"].round(2).to_string())