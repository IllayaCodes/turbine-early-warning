import numpy as np
import pandas as pd

from src.turbine_ew.models import (
    error_summary,
    fit_model,
    make_boosting,
    make_linear,
    make_mean,
    residuals,
)

FEATURES = ["wind_speed", "power"]


def make_frame(seed, n=600):
    rng = np.random.default_rng(seed)
    wind = rng.uniform(3, 15, n)
    power = rng.uniform(0, 2000, n)
    temp = 30 + (wind - 9) ** 2 + 0.005 * power
    index = pd.date_range("2017-01-01", periods=n, freq="10min")
    return pd.DataFrame({"wind_speed": wind, "power": power, "temp": temp}, index=index)


def test_mean_model_predicts_the_training_average():
    train, test = make_frame(0), make_frame(1)
    model = fit_model(make_mean(), train, FEATURES, "temp")
    res = residuals(model, test, FEATURES, "temp")
    assert np.allclose(res, test["temp"] - train["temp"].mean())


def test_boosting_beats_linear_on_a_curved_relation():
    train, test = make_frame(0), make_frame(1)
    linear = fit_model(make_linear(), train, FEATURES, "temp")
    boost = fit_model(make_boosting(), train, FEATURES, "temp")
    mae_linear = error_summary(residuals(linear, test, FEATURES, "temp"))["mae"]
    mae_boost = error_summary(residuals(boost, test, FEATURES, "temp"))["mae"]
    assert mae_boost < mae_linear / 2


def test_boosting_gives_the_same_answer_every_run():
    train, test = make_frame(0), make_frame(1)
    a = fit_model(make_boosting(), train, FEATURES, "temp").predict(test[FEATURES])
    b = fit_model(make_boosting(), train, FEATURES, "temp").predict(test[FEATURES])
    assert np.allclose(a, b)