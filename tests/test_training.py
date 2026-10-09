import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.turbine_ew.training import save_artifacts

CFG = {
    "model": {"features": ["wind_speed", "power"], "targets": ["gen_bearing_rear_temp"]},
    "event": {
        "turbine": 1,
        "fault_start": "2021-02-20 00:00:00",
        "split": {
            "validation": ["2020-07-01", "2020-09-30"],
            "test": ["2020-11-01", "2021-03-31"],
        },
    },
    "alerts": {
        "min_other_turbines": 3,
        "k": 4.0,
        "floor_c": 2.0,
        "min_validation_days": 30,
        "consecutive_days": 3,
        "window_days": 30,
    },
}


def make_daily():
    idx = pd.date_range("2020-07-01", "2021-03-31", freq="D")
    rng = np.random.default_rng(0)
    data = {t: pd.Series(rng.normal(0, 0.1, len(idx)), index=idx) for t in range(1, 5)}
    data[1].loc[pd.date_range("2021-02-01", "2021-02-05")] += 5.0
    return {"gen_bearing_rear_temp": data}


def make_model():
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]])
    return LinearRegression().fit(x, np.array([1.0, 2.0, 4.0]))


def test_artifacts_are_saved_and_can_be_reloaded(tmp_path):
    models = {(1, "gen_bearing_rear_temp"): make_model()}
    thresholds, alerts = save_artifacts(CFG, models, make_daily(), "linear", "event", tmp_path)
    saved = joblib.load(tmp_path / "models.joblib")
    assert saved["features"] == ["wind_speed", "power"]
    assert saved["model_name"] == "linear"
    assert (1, "gen_bearing_rear_temp") in saved["models"]
    assert (tmp_path / "thresholds.csv").exists()
    assert len(pd.read_csv(tmp_path / "alerts.csv")) == 1
    assert len(alerts) == 1


def test_a_reloaded_model_predicts_like_the_original(tmp_path):
    model = make_model()
    save_artifacts(CFG, {(1, "gen_bearing_rear_temp"): model}, make_daily(), "linear", "event", tmp_path)
    reloaded = joblib.load(tmp_path / "models.joblib")["models"][(1, "gen_bearing_rear_temp")]
    x = np.array([[1.5, 2.0]])
    assert np.allclose(model.predict(x), reloaded.predict(x))