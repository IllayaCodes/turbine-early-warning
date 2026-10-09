import pandas as pd
import yaml

from src.turbine_ew.data import load_turbine_years
from src.turbine_ew.models import (
    error_summary,
    fit_model,
    make_boosting,
    make_linear,
    make_mean,
    residuals,
)
from src.turbine_ew.preprocess import clean
from src.turbine_ew.split import time_split

with open("configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

d = cfg["data"]
ev = cfg["event"]
features = cfg["model"]["features"]
targets = cfg["model"]["targets"]
makers = {"mean": make_mean, "linear": make_linear, "boosting": make_boosting}

rows = []
for turbine in range(1, 7):
    df = load_turbine_years(d["raw_dir"], ev["years"], turbine, cfg["signals"], d["skip_rows"])
    out, _ = clean(df, cfg["cleaning"])
    parts = time_split(out, ev["split"])
    for target in targets:
        for name, maker in makers.items():
            model = fit_model(maker(), parts["train"], features, target)
            for block in ["validation", "test"]:
                s = error_summary(residuals(model, parts[block], features, target))
                rows.append({"turbine": turbine, "target": target, "model": name, "block": block, **s})
    print("done turbine", turbine, flush=True)

table = pd.DataFrame(rows)
table.to_csv("docs/errors_event_period.csv", index=False)
summary = table.groupby(["target", "model", "block"])[["mae", "rmse", "bias"]].mean().round(2)
print()
print("mean over the 6 turbines (degrees C), event period")
print(summary.to_string())