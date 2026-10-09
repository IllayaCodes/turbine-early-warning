from pathlib import Path

import joblib

from src.turbine_ew.alerts import daily_mean
from src.turbine_ew.data import load_turbine_years
from src.turbine_ew.models import fit_model, residuals
from src.turbine_ew.pipeline import MAKERS, event_alerts
from src.turbine_ew.preprocess import clean
from src.turbine_ew.split import time_split


def train_models(cfg, model_name, section="event", turbines=range(1, 7)):
    """Fit one model per turbine and target on the section's train block only.

    Returns (models, daily). models[(turbine, target)] is a fitted model and
    daily[target][turbine] is the daily mean residual over all the section's years.
    """
    d = cfg["data"]
    sec = cfg[section]
    features = cfg["model"]["features"]
    targets = cfg["model"]["targets"]
    min_rows = cfg["alerts"]["min_rows_per_day"]
    models = {}
    daily = {target: {} for target in targets}
    for turbine in turbines:
        df = load_turbine_years(d["raw_dir"], sec["years"], turbine, cfg["signals"], d["skip_rows"])
        out, _ = clean(df, cfg["cleaning"])
        parts = time_split(out, sec["split"])
        for target in targets:
            model = fit_model(MAKERS[model_name](), parts["train"], features, target)
            models[(turbine, target)] = model
            daily[target][turbine] = daily_mean(residuals(model, out, features, target), min_rows)
        print("trained turbine", turbine, flush=True)
    return models, daily


def save_artifacts(cfg, models, daily, model_name, section, out_dir):
    """Save the models, the thresholds and the alerts. Returns (thresholds, alerts)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    thresholds, alerts = event_alerts(cfg, daily, section=section)
    joblib.dump(
        {
            "models": models,
            "features": cfg["model"]["features"],
            "targets": cfg["model"]["targets"],
            "model_name": model_name,
            "section": section,
        },
        out / "models.joblib",
    )
    thresholds.to_csv(out / "thresholds.csv", index=False)
    alerts.to_csv(out / "alerts.csv", index=False)
    return thresholds, alerts