import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

from src.turbine_ew.api import create_app

FEATURES = ["wind_speed", "power", "rotor_speed", "gearbox_speed", "ambient_temp"]
TARGETS = ["gear_oil_temp", "gen_bearing_rear_temp"]
CLEANING = {"min_wind_speed": 3.0, "max_wind_speed": 25.0, "min_power": 0.0, "min_rotor_speed": 5.0}
READING = {
    "turbine": 1,
    "wind_speed": 8.0,
    "power": 900.0,
    "rotor_speed": 13.0,
    "gearbox_speed": 1300.0,
    "ambient_temp": 10.0,
}


def make_model(intercept):
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(10, len(FEATURES))), columns=FEATURES)
    return LinearRegression().fit(x, np.full(10, intercept))


@pytest.fixture
def client():
    models = {
        (1, "gear_oil_temp"): make_model(55.0),
        (1, "gen_bearing_rear_temp"): make_model(40.0),
        (2, "gear_oil_temp"): make_model(56.0),
        (2, "gen_bearing_rear_temp"): make_model(41.0),
    }
    artifacts = {
        "models": models,
        "features": FEATURES,
        "targets": TARGETS,
        "model_name": "linear",
        "section": "event",
    }
    thresholds = pd.DataFrame({
        "turbine": [1, 2],
        "target": ["gen_bearing_rear_temp", "gen_bearing_rear_temp"],
        "threshold_C": [2.1, 2.0],
    })
    alerts = pd.DataFrame({
        "turbine": [2, 1],
        "target": ["gen_bearing_rear_temp", "gen_bearing_rear_temp"],
        "start": ["2021-01-19", "2020-12-09"],
        "alert_date": ["2021-01-21", "2020-12-11"],
        "end": ["2021-01-23", "2020-12-12"],
        "days": [5, 4],
        "lead_days": [15.0, np.nan],
    })
    return TestClient(create_app(artifacts, thresholds, alerts, CLEANING))


def test_health_lists_the_model_and_turbines(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "linear"
    assert body["turbines"] == [1, 2]


def test_predict_returns_the_expected_temperatures(client):
    body = client.post("/predict", json=READING).json()
    assert body["in_operating_range"] is True
    assert body["expected"]["gear_oil_temp"] == pytest.approx(55.0)
    assert body["expected"]["gen_bearing_rear_temp"] == pytest.approx(40.0)
    assert body["residual"] is None


def test_predict_residual_is_actual_minus_expected(client):
    reading = dict(READING, actual={"gen_bearing_rear_temp": 46.0})
    body = client.post("/predict", json=reading).json()
    assert body["residual"]["gen_bearing_rear_temp"] == pytest.approx(6.0)


def test_predict_outside_the_operating_range_gives_no_residual(client):
    reading = dict(READING, wind_speed=2.0, actual={"gear_oil_temp": 60.0})
    body = client.post("/predict", json=reading).json()
    assert body["in_operating_range"] is False
    assert body["residual"] is None


def test_predict_unknown_turbine_is_404(client):
    response = client.post("/predict", json=dict(READING, turbine=9))
    assert response.status_code == 404


def test_predict_unknown_target_in_actual_is_422(client):
    response = client.post("/predict", json=dict(READING, actual={"hub_temp": 30.0}))
    assert response.status_code == 422


def test_predict_rejects_impossible_values(client):
    response = client.post("/predict", json=dict(READING, wind_speed=-5.0))
    assert response.status_code == 422


def test_alerts_returns_all_with_thresholds_and_nulls(client):
    body = client.get("/alerts").json()
    assert body["count"] == 2
    by_turbine = {a["turbine"]: a for a in body["alerts"]}
    assert by_turbine[2]["lead_days"] == 15
    assert by_turbine[1]["lead_days"] is None
    assert by_turbine[1]["threshold_C"] == pytest.approx(2.1)
    assert by_turbine[2]["start"] == "2021-01-19"


def test_alerts_filter_by_turbine(client):
    body = client.get("/alerts", params={"turbine": 2}).json()
    assert body["count"] == 1
    assert body["alerts"][0]["turbine"] == 2


def test_alerts_filter_by_dates(client):
    assert client.get("/alerts", params={"until": "2020-12-31"}).json()["count"] == 1
    assert client.get("/alerts", params={"since": "2021-01-24"}).json()["count"] == 0
    assert client.get("/alerts", params={"since": "2021-01-23"}).json()["count"] == 1


def test_alerts_rejects_a_bad_date(client):
    assert client.get("/alerts", params={"since": "not-a-date"}).status_code == 422