"""Trend-following signal based on moving average crossover."""

from __future__ import annotations

import pandas as pd


def generate_signal(close: pd.Series, fast: int = 20, slow: int = 80) -> pd.Series:
    f = close.rolling(fast, min_periods=1).mean()
    s = close.rolling(slow, min_periods=1).mean()
    signal = (f > s).astype(float)
    return signal
