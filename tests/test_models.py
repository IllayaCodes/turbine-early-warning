import numpy as np
import pandas as pd
import pytest

from src.turbine_ew.models import error_summary, fit_model, make_linear, residuals

FEATURES = ["wind_speed", "power"]


def make_frame(offset=0.0, n=200):
    rng = np.random.default_rng(0)
    wind = rng.uniform(3, 15, n)
    power = rng.uniform(0, 2000, n)
    temp = 20 + 1.5 * wind + 0.01 * power + offset
    index = pd.date_range("2017-01-01", periods=n, freq="10min")
    return pd.DataFrame({"wind_speed": wind, "power": power, "temp": temp}, index=index)


def test_model_recovers_an_exact_relation():
    df = make_frame()
    model = fit_model(make_linear(), df, FEATURES, "temp")
    assert residuals(model, df, FEATURES, "temp").abs().max() < 1e-6


def test_hotter_than_normal_gives_a_positive_residual():
    model = fit_model(make_linear(), make_frame(), FEATURES, "temp")
    hot = make_frame(offset=5.0)
    assert residuals(model, hot, FEATURES, "temp").mean() == pytest.approx(5.0, abs=1e-6)


def test_target_cannot_be_its_own_feature():
    with pytest.raises(ValueError):
        fit_model(make_linear(), make_frame(), ["wind_speed", "temp"], "temp")


def test_error_summary_values():
    res = pd.Series([1.0, -1.0, 3.0, -3.0])
    s = error_summary(res)
    assert s["mae"] == pytest.approx(2.0)
    assert s["rmse"] == pytest.approx(5 ** 0.5)
    assert s["bias"] == pytest.approx(0.0)