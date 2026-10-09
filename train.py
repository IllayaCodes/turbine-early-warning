import yaml

from src.turbine_ew.training import save_artifacts, train_models

with open("configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

s = cfg["serving"]
models, daily = train_models(cfg, s["model"], s["section"])
thresholds, alerts = save_artifacts(cfg, models, daily, s["model"], s["section"], s["models_dir"])
print("saved to", s["models_dir"])
print("models:", len(models), " thresholds:", len(thresholds), " alerts:", len(alerts))