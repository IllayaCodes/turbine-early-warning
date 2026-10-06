from pathlib import Path

import yaml

from src.turbine_ew.data import load_turbine_file

with open("configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

path = next(Path("data/raw/kelmarsh2017").glob("Turbine_Data_Kelmarsh_1_*.csv"))
df = load_turbine_file(path, cfg["signals"], cfg["data"]["skip_rows"])
print(df.shape)
print(df.head())
print(df.describe().round(1).T)