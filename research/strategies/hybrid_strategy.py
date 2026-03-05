"""Hybrid strategy blending trend and grid positioning."""

from __future__ import annotations

import pandas as pd

from strategies.grid_strategy import generate_signal as grid_signal
from strategies.trend_following import generate_signal as trend_signal


def generate_signal(close: pd.Series, grid_spacing: float = 0.01, reserve_weight: float = 0.2) -> pd.Series:
    g = grid_signal(close, grid_spacing=grid_spacing)
    t = trend_signal(close)
    mixed = (1 - reserve_weight) * (0.5 * g + 0.5 * t)
    return mixed.clip(-1, 1)
