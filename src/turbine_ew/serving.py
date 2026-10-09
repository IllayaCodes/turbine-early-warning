from pathlib import Path

import joblib
import pandas as pd


def load_artifacts(models_dir):
    """Load the saved models, thresholds and alerts written by train.py."""
    folder = Path(models_dir)
    artifacts = joblib.load(folder / "models.joblib")
    thresholds = pd.read_csv(folder / "thresholds.csv")
    alerts = pd.read_csv(folder / "alerts.csv")
    return artifacts, thresholds, alerts


def operating_ok(row, cleaning):
    """True when the inputs are inside the range the models were trained on."""
    return (
        cleaning["min_wind_speed"] <= row["wind_speed"] <= cleaning["max_wind_speed"]
        and row["power"] > cleaning["min_power"]
        and row["rotor_speed"] >= cleaning["min_rotor_speed"]
    )


def has_turbine(artifacts, turbine):
    return any(t == turbine for t, _ in artifacts["models"])


def predict_turbine(artifacts, turbine, row):
    """Expected temperature per target for one turbine and one row of signals."""
    features = artifacts["features"]
    x = pd.DataFrame([[row[f] for f in features]], columns=features)
    return {
        target: float(artifacts["models"][(turbine, target)].predict(x)[0])
        for target in artifacts["targets"]
    }


def select_alerts(alerts, thresholds, turbine=None, target=None, since=None, until=None):
    """Alerts joined with their thresholds, filtered, as a list of plain dicts."""
    df = alerts.merge(thresholds, on=["turbine", "target"], how="left")
    for col in ("start", "alert_date", "end"):
        df[col] = pd.to_datetime(df[col])
    if turbine is not None:
        df = df[df["turbine"] == turbine]
    if target is not None:
        df = df[df["target"] == target]
    if since is not None:
        df = df[df["end"] >= pd.Timestamp(since)]
    if until is not None:
        df = df[df["start"] <= pd.Timestamp(until)]
    df = df.copy()
    for col in ("start", "alert_date", "end"):
        df[col] = df[col].dt.strftime("%Y-%m-%d")
    df = df.astype(object).where(df.notna(), None)
    return df.to_dict(orient="records")