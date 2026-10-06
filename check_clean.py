from pathlib import Path

import yaml

from src.turbine_ew.data import load_turbine_file
from src.turbine_ew.preprocess import clean

with open("configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

path = next(Path("data/raw/kelmarsh2017").glob("Turbine_Data_Kelmarsh_1_*.csv"))
df = load_turbine_file(path, cfg["signals"], cfg["data"]["skip_rows"])
out, report = clean(df, cfg["cleaning"])
print(report)
print(out.describe().round(1).T)