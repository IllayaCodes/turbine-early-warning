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
target = "gen_bearing_rear_temp"
subject = ev["turbine"]

daily = {}
for turbine in range(1, 7):
    df = load_turbine_years(d["raw_dir"], ev["years"], turbine, cfg["signals"], d["skip_rows"])
    out, _ = clean(df, cfg["cleaning"])
    parts = time_split(out, ev["split"])
    model = fit_model(make_boosting(), parts["train"], features, target)
    daily[turbine] = residuals(model, out, features, target).resample("D").mean()
    if turbine == subject:
        rows = out[target].resample("D").count()
        actual = out[target].resample("D").mean()
        power = out["power"].resample("D").mean()
    print("done turbine", turbine, flush=True)

others = pd.concat([daily[t] for t in range(1, 7) if t != subject], axis=1)
table = pd.DataFrame({
    "rows": rows,
    "actual_C": actual,
    "power_kW": power,
    "resid_raw": daily[subject],
    "others_median": others.median(axis=1),
})
table["resid_relative"] = table["resid_raw"] - table["others_median"]
print("turbine", subject, "rear generator bearing, daily (rows = operating 10-minute rows, max 144)")
print(table.loc["2021-01-10":"2021-02-06"].round(1).to_string())