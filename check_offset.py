import pandas as pd
import yaml

from src.turbine_ew.alerts import fleet_relative
from src.turbine_ew.pipeline import daily_residuals

with open("configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

daily = daily_residuals(cfg, ["boosting"])
target = "gen_bearing_rear_temp"
rel = fleet_relative(pd.DataFrame(daily["boosting"][target]), cfg["alerts"]["min_other_turbines"])
monthly = rel.resample("MS").median().loc["2019-07-01":"2021-06-30"]
monthly.index = monthly.index.strftime("%Y-%m")
print()
print(target, "- monthly median of (turbine residual minus median of the others), boosting")
print("training block", cfg["event"]["split"]["train"], "validation block", cfg["event"]["split"]["validation"])
print(monthly.round(1).to_string())