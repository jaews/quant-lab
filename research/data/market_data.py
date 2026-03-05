"""Market data helpers for local research workflows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if "close" not in df.columns:
        raise ValueError("Input data must include a 'close' column")
    return df.sort_values("timestamp").reset_index(drop=True)
