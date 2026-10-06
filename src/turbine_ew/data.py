from pathlib import Path

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


def load_turbine_years(raw_dir, years, turbine, signals, skip_rows=9):
    """Load one turbine across several yearly folders and join them in time order."""
    frames = []
    for year in years:
        folder = Path(raw_dir) / f"kelmarsh{year}"
        files = sorted(folder.glob(f"Turbine_Data_Kelmarsh_{turbine}_*.csv"))
        if len(files) != 1:
            raise FileNotFoundError(f"expected one file for turbine {turbine} in {folder}, found {files}")
        frames.append(load_turbine_file(files[0], signals, skip_rows))
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="first")]