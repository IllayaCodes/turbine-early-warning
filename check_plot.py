import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
for ax, target in zip(axes, cfg["model"]["targets"]):
    model = fit_model(make_boosting(), parts["train"], features, target)
    res = residuals(model, out, features, target)
    daily = res.resample("D").mean()
    ax.plot(daily.index, daily, color="lightgray", linewidth=0.8, label="daily average")
    ax.plot(daily.index, daily.rolling(14, min_periods=7).mean(), color="tab:red", linewidth=1.6, label="14-day average")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(pd.Timestamp(cfg["split"]["validation"][0]), color="tab:blue", linestyle="--")
    ax.axvline(pd.Timestamp(cfg["split"]["test"][0]), color="tab:green", linestyle="--")
    ax.set_ylabel(target + " (C)")
    ax.set_ylim(-4, 8)
axes[0].legend(loc="upper left")
axes[0].set_title("Turbine 1: actual minus predicted temperature (boosting, trained on 2016-05 to 2017-07)")
fig.tight_layout()
fig.savefig("docs/drift_turbine1.png", dpi=110)
print("saved docs/drift_turbine1.png")