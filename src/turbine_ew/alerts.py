import pandas as pd


def daily_mean(res, min_rows):
    """Daily mean of a residual series. Days with fewer than min_rows rows become NaN."""
    grouped = res.resample("D")
    return grouped.mean().where(grouped.count() >= min_rows)


def fleet_relative(daily, min_others=3):
    """Subtract, for each turbine and day, the median of the other turbines.

    daily has one column per turbine. Where fewer than min_others other turbines
    have a value that day, the result is NaN.
    """
    out = pd.DataFrame(index=daily.index, columns=daily.columns, dtype=float)
    for col in daily.columns:
        others = daily.drop(columns=col)
        median = others.median(axis=1)
        enough = others.notna().sum(axis=1) >= min_others
        out[col] = (daily[col] - median).where(enough)
    return out


def spread(x):
    """Robust spread: 1.4826 times the median absolute deviation, ignoring NaN."""
    x = x.dropna()
    return float(1.4826 * (x - x.median()).abs().median())


def threshold(valid, k, floor, min_days):
    """Alert threshold in degrees C, from healthy validation data only."""
    x = valid.dropna()
    if len(x) < min_days:
        raise ValueError(f"only {len(x)} valid days to set the threshold, need {min_days}")
    return max(k * spread(x), floor)


def find_alerts(rel, thr, n_days):
    """Runs of at least n_days consecutive calendar days with rel above thr.

    Returns a list of dicts with start, alert_date (the day the n-th consecutive
    day is reached, which is when a live system could first have fired) and end.
    Days that are NaN count as not above the threshold.
    """
    if len(rel) > 1 and not (rel.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("rel must have exactly one row per calendar day")
    above = rel > thr
    run_id = (above != above.shift(fill_value=False)).cumsum()
    alerts = []
    for _, run in above.groupby(run_id):
        if run.iloc[0] and len(run) >= n_days:
            days = run.index
            alerts.append({"start": days[0], "alert_date": days[n_days - 1], "end": days[-1]})
    return alerts


def lead_days(alert_date, fault_start, window_days):
    """Days from the alert to the fault if the fault follows within the window, else None."""
    delta = (pd.Timestamp(fault_start).normalize() - pd.Timestamp(alert_date).normalize()).days
    if 0 <= delta <= window_days:
        return delta
    return None

def merge_episodes(alerts, max_gap_days):
    """Merge alerts on the same turbine and target when the gap is at most max_gap_days.

    alerts needs the columns turbine, target, start, end (as Timestamps) and lead_days.
    Returns one row per episode with start, end, n_alerts and lead_days.
    """
    columns = ["turbine", "target", "start", "end", "n_alerts", "lead_days"]
    rows = []
    ordered = alerts.sort_values(["turbine", "target", "start"])
    for (turbine, target), group in ordered.groupby(["turbine", "target"]):
        cur = None
        for _, a in group.iterrows():
            if cur is not None and (a["start"] - cur["end"]).days <= max_gap_days:
                cur["end"] = max(cur["end"], a["end"])
                cur["n_alerts"] += 1
                if pd.isna(cur["lead_days"]):
                    cur["lead_days"] = a["lead_days"]
            else:
                if cur is not None:
                    rows.append(cur)
                cur = {
                    "turbine": turbine,
                    "target": target,
                    "start": a["start"],
                    "end": a["end"],
                    "n_alerts": 1,
                    "lead_days": a["lead_days"],
                }
        if cur is not None:
            rows.append(cur)
    return pd.DataFrame(rows, columns=columns)