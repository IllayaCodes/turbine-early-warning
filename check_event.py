import pandas as pd
import yaml

from src.turbine_ew.data import load_turbine_years
from src.turbine_ew.models import fit_model, make_boosting, residuals
from src.turbine_ew.preprocess import clean
from src.turbine_ew.split import time_split

with open("configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

d = cfg["data"]
ev = cfg["event"]
features = cfg["model"]["features"]
targets = cfg["model"]["targets"]
subject = ev["turbine"]

daily = {}
for turbine in range(1, 7):
    df = load_turbine_years(d["raw_dir"], ev["years"], turbine, cfg["signals"], d["skip_rows"])
    out, _ = clean(df, cfg["cleaning"])
    parts = time_split(out, ev["split"])
    for target in targets:
        model = fit_model(make_boosting(), parts["train"], features, target)
        res = residuals(model, out, features, target)
        daily[(turbine, target)] = res.resample("D").mean()
    print("done turbine", turbine, flush=True)

raw = {}
rel = {}
for target in targets:
    others = pd.concat([daily[(t, target)] for t in range(1, 7) if t != subject], axis=1)
    raw[target] = daily[(subject, target)]
    rel[target] = daily[(subject, target)] - others.median(axis=1)

weekly_raw = pd.DataFrame(raw).resample("W").mean()
weekly_rel = pd.DataFrame(rel).resample("W").mean()

val = ev["split"]["validation"]
spread = pd.DataFrame({
    "raw": weekly_raw.loc[val[0]:val[1]].std(),
    "fleet_relative": weekly_rel.loc[val[0]:val[1]].std(),
})
print()
print("normal weekly spread (std of weekly means over the validation block):")
print(spread.round(2).to_string())

start, end = "2020-11-01", "2021-03-21"
print()
print("turbine", subject, "- fault starts", ev["fault_start"])
print("weekly mean residual, raw:")
print(weekly_raw.loc[start:end].round(2).to_string())
print("weekly mean residual, minus median of the other five turbines:")
print(weekly_rel.loc[start:end].round(2).to_string())