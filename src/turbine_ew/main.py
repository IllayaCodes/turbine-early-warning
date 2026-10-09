import os

import yaml

from src.turbine_ew.api import create_app
from src.turbine_ew.serving import load_artifacts

with open(os.environ.get("TURBINE_CONFIG", "configs/config.yaml"), encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

models_dir = os.environ.get("TURBINE_MODELS", cfg["serving"]["models_dir"])
artifacts, thresholds, alerts = load_artifacts(models_dir)
app = create_app(artifacts, thresholds, alerts, cfg["cleaning"])