import numpy as np
from sklearn.linear_model import LinearRegression


def make_linear():
    return LinearRegression()


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