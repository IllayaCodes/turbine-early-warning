import pandas as pd


def plausible_mask(df, limits):
    """True where every limited signal is inside its [low, high] range."""
    mask = pd.Series(True, index=df.index)
    for name, (low, high) in limits.items():
        mask &= df[name].between(low, high)
    return mask


def operating_mask(df, cleaning):
    """True where the turbine is producing power in its normal wind range."""
    return (
        df["wind_speed"].between(cleaning["min_wind_speed"], cleaning["max_wind_speed"])
        & (df["power"] > cleaning["min_power"])
        & (df["rotor_speed"] >= cleaning["min_rotor_speed"])
    )


def clean(df, cleaning):
    """Keep only complete, plausible, operating rows. Never fills missing values."""
    has_all = df.notna().all(axis=1)
    in_range = plausible_mask(df, cleaning["limits"])
    operating = operating_mask(df, cleaning)
    keep = has_all & in_range & operating
    report = {
        "rows_in": len(df),
        "missing": int((~has_all).sum()),
        "implausible": int((has_all & ~in_range).sum()),
        "not_operating": int((has_all & in_range & ~operating).sum()),
        "rows_out": int(keep.sum()),
    }
    return df[keep], report