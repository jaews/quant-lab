"""Toy grid strategy implementation for research experiments."""

from __future__ import annotations

import pandas as pd


def generate_signal(close: pd.Series, grid_spacing: float = 0.01) -> pd.Series:
    ma = close.rolling(24, min_periods=1).mean()
    deviation = (close - ma) / ma
    signal = (-deviation / grid_spacing).clip(-1, 1)
    return signal.fillna(0.0)
