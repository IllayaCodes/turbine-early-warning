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
features = cfg["model"]["features"]
targets = cfg["model"]["targets"]

smooth = {}
rows = []
for turbine in range(1, 7):
    df = load_turbine_years(d["raw_dir"], d["years"], turbine, cfg["signals"], d["skip_rows"])
    out, _ = clean(df, cfg["cleaning"])
    parts = time_split(out, cfg["split"])
    for target in targets:
        model = fit_model(make_boosting(), parts["train"], features, target)
        res = residuals(model, out, features, target)
        roll = res.resample("D").mean().rolling(14, min_periods=7).mean()
        smooth[(turbine, target)] = roll
        rows.append({
            "turbine": turbine,
            "target": target,
            "bias_val": residuals(model, parts["validation"], features, target).mean(),
            "bias_test": residuals(model, parts["test"], features, target).mean(),
            "peak_14d": roll.max(),
            "peak_date": roll.idxmax().date(),
        })
    print("done turbine", turbine, flush=True)

fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
for ax, target in zip(axes, targets):
    for turbine in range(1, 7):
        s = smooth[(turbine, target)]
        ax.plot(s.index, s, linewidth=1.2, label="turbine " + str(turbine))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvspan(pd.Timestamp("2017-11-11"), pd.Timestamp("2017-11-16"), color="orange", alpha=0.5)
    ax.axvline(pd.Timestamp(cfg["split"]["test"][0]), color="tab:green", linestyle="--")
    ax.set_ylabel(target + " (C)")
    ax.set_ylim(-4, 8)
axes[0].legend(loc="upper left", ncol=6, fontsize=8)
axes[0].set_title("14-day average of actual minus predicted, all six turbines")
fig.tight_layout()
fig.savefig("docs/fleet_drift.png", dpi=110)
print(pd.DataFrame(rows).round(2).to_string(index=False))