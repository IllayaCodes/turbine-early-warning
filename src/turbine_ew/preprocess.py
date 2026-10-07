import pandas as pd


def plausible_mask(df, limits):
    """True where every limited signal is inside its [low, high] range."""
    mask = pd.Series(True, index=df.index)
    for name, (low, high) in limits.items():
        mask &= df[name].between(low, high)
    return mask


def producing_mask(df, cleaning):
    """True where the turbine is turning and delivering power."""
    return (df["power"] > cleaning["min_power"]) & (df["rotor_speed"] >= cleaning["min_rotor_speed"])


def operating_mask(df, cleaning):
    """True where the turbine is producing power in its normal wind range."""
    in_wind_range = df["wind_speed"].between(cleaning["min_wind_speed"], cleaning["max_wind_speed"])
    return in_wind_range & producing_mask(df, cleaning)


def warmup_mask(df, cleaning):
    """True for rows shortly after a long stop, while components are still heating up.

    A long stop is a run of at least min_stop_rows rows without production
    (missing data counts as not producing). The next warmup_rows rows are flagged.
    """
    stopped = ~producing_mask(df, cleaning)
    change = stopped.astype(int).diff().fillna(1).ne(0)
    run_id = change.cumsum()
    run_length = stopped.astype(int).groupby(run_id).transform("sum")
    long_stop = stopped & (run_length >= cleaning["min_stop_rows"])
    window = cleaning["warmup_rows"] + 1
    recent = long_stop.astype(int).rolling(window, min_periods=1).max()
    return recent.astype(bool)


def clean(df, cleaning):
    """Keep only complete, plausible, operating, warmed-up rows. Never fills missing values."""
    has_all = df.notna().all(axis=1)
    in_range = plausible_mask(df, cleaning["limits"])
    operating = operating_mask(df, cleaning)
    warming = warmup_mask(df, cleaning)
    keep = has_all & in_range & operating & ~warming
    report = {
        "rows_in": len(df),
        "missing": int((~has_all).sum()),
        "implausible": int((has_all & ~in_range).sum()),
        "not_operating": int((has_all & in_range & ~operating).sum()),
        "warming_up": int((has_all & in_range & operating & warming).sum()),
        "rows_out": int(keep.sum()),
    }
    return df[keep], report