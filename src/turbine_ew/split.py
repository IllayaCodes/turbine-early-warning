import pandas as pd

NAMES = ["train", "validation", "test"]


def time_split(df, split):
    """Split a time-indexed table into train, validation and test by date ranges.

    The ranges must be in time order and must not overlap. Rows are never shuffled.
    """
    ranges = []
    for name in NAMES:
        start = pd.Timestamp(split[name][0])
        end = pd.Timestamp(split[name][1]) + pd.Timedelta(days=1)
        if not start < end:
            raise ValueError(f"{name} range ends before it starts")
        ranges.append((start, end))
    for (_, end_a), (start_b, _) in zip(ranges, ranges[1:]):
        if end_a > start_b:
            raise ValueError("split ranges overlap or are out of time order")
    return {
        name: df[(df.index >= start) & (df.index < end)]
        for name, (start, end) in zip(NAMES, ranges)
    }