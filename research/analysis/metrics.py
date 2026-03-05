"""Performance metrics from equity curves and trade summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.drawdown import average_drawdown_duration, drawdown_series

TRADING_DAYS = 252


def daily_returns(equity: pd.Series) -> pd.Series:
    eq = pd.Series(equity).astype(float)
    returns = eq.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return returns


def cagr(equity: pd.Series) -> float:
    eq = pd.Series(equity).astype(float)
    if len(eq) < 2 or eq.iloc[0] <= 0:
        return 0.0
    years = len(eq) / TRADING_DAYS
    if years <= 0:
        return 0.0
    return float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1)


def sharpe_ratio(equity: pd.Series, risk_free_rate: float = 0.0) -> float:
    rets = daily_returns(equity)
    excess = rets - risk_free_rate / TRADING_DAYS
    std = excess.std(ddof=0)
    if std == 0:
        return 0.0
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / std)


def sortino_ratio(equity: pd.Series, risk_free_rate: float = 0.0) -> float:
    rets = daily_returns(equity)
    excess = rets - risk_free_rate / TRADING_DAYS
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=0)
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / downside_std)


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown_series(equity).min())


def calmar_ratio(equity: pd.Series) -> float:
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    return float(cagr(equity) / mdd)


def profit_factor(trades: pd.DataFrame) -> float:
    if trades is None or trades.empty or "pnl" not in trades.columns:
        return 0.0
    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def win_rate(trades: pd.DataFrame) -> float:
    if trades is None or trades.empty or "pnl" not in trades.columns:
        return 0.0
    return float((trades["pnl"] > 0).mean())


def compute_metrics(equity: pd.Series, trades: pd.DataFrame | None = None) -> dict:
    metrics = {
        "CAGR": cagr(equity),
        "Sharpe": sharpe_ratio(equity),
        "Sortino": sortino_ratio(equity),
        "MaxDD": max_drawdown(equity),
        "Calmar": calmar_ratio(equity),
        "AvgDDDuration": average_drawdown_duration(equity),
    }
    if trades is not None:
        metrics["ProfitFactor"] = profit_factor(trades)
        metrics["WinRate"] = win_rate(trades)
    return metrics
