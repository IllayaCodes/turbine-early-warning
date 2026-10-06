import pandas as pd


def find_column(columns, base_name):
    """Find the single column whose name is base_name followed by a unit in brackets."""
    matches = [c for c in columns if c.startswith(base_name + " (")]
    if len(matches) != 1:
        raise ValueError(f"expected one column for {base_name!r}, found {matches}")
    return matches[0]


def load_turbine_file(path, signals, skip_rows=9):
    """Load one Kelmarsh Turbine_Data csv and keep only the configured signals."""
    df = pd.read_csv(path, skiprows=skip_rows, na_values=["NaN"], low_memory=False)
    df = df.rename(columns={df.columns[0]: "time"})
    df["time"] = pd.to_datetime(df["time"])
    out = pd.DataFrame({"time": df["time"]})
    for short_name, base_name in signals.items():
        out[short_name] = df[find_column(df.columns, base_name)]
    return out.set_index("time").sort_index()