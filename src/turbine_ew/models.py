import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression


def make_mean():
    """Reference model that always predicts the training average."""
    return DummyRegressor(strategy="mean")


def make_linear():
    return LinearRegression()


def make_boosting(random_state=0):
    """Gradient boosting. Early stopping is off so no hidden random split is made."""
    return HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.1,
        max_depth=6,
        early_stopping=False,
        random_state=random_state,
    )


def fit_model(model, train_df, features, target):
    """Fit on the training rows only. The target may never be one of its own inputs."""
    if target in features:
        raise ValueError(f"target {target!r} is also listed as a feature")
    model.fit(train_df[features], train_df[target])
    return model


def residuals(model, df, features, target):
    """Actual minus predicted. Positive means hotter than the model expects."""
    predicted = model.predict(df[features])
    return df[target] - predicted


def error_summary(res):
    """Typical size (mae), big-miss-sensitive size (rmse) and average direction (bias)."""
    return {
        "mae": float(res.abs().mean()),
        "rmse": float(np.sqrt((res ** 2).mean())),
        "bias": float(res.mean()),
    }