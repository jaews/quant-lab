"""Drawdown analytics utilities."""

from __future__ import annotations

import pandas as pd


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Return underwater series from an equity curve."""
    eq = pd.Series(equity).astype(float)
    rolling_peak = eq.cummax()
    dd = eq / rolling_peak - 1.0
    return dd.fillna(0.0)


def drawdown_duration(equity: pd.Series) -> pd.Series:
    """Count consecutive periods spent below high-water mark."""
    dd = drawdown_series(equity)
    in_drawdown = dd < 0
    durations = []
    current = 0
    for flag in in_drawdown:
        if flag:
            current += 1
        else:
            current = 0
        durations.append(current)
    return pd.Series(durations, index=dd.index, dtype=int)


def average_drawdown_duration(equity: pd.Series) -> float:
    """Average drawdown duration across completed drawdown episodes."""
    durations = drawdown_duration(equity)
    completed = []
    current_max = 0
    for value in durations:
        current_max = max(current_max, int(value))
        if value == 0 and current_max > 0:
            completed.append(current_max)
            current_max = 0
    if current_max > 0:
        completed.append(current_max)
    return float(pd.Series(completed).mean()) if completed else 0.0
