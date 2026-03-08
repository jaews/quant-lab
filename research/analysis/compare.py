"""Strategy comparison and leaderboard generation."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from analysis.metrics import compute_metrics
from engine.io import list_runs, load_run

LOGGER = logging.getLogger(__name__)

DEFAULT_COLUMNS = [
    "Strategy",
    "Symbol",
    "Timeframe",
    "CAGR",
    "Sharpe",
    "Sortino",
    "Calmar",
    "MaxDD",
]


def _collect_rows(root: str | Path = "results/runs") -> list[dict]:
    rows: list[dict] = []
    for run_path in list_runs(root):
        try:
            run = load_run(run_path)
        except Exception as exc:
            LOGGER.warning("Skipping run '%s': %s", run_path.name, exc)
            continue
        m = compute_metrics(run["equity"]["equity"], run["trades"])
        rows.append(
            {
                "Run": run["name"],
                "Strategy": run["meta"].get("strategy_name", "unknown"),
                "Symbol": run["meta"].get("symbol", "unknown"),
                "Timeframe": run["meta"].get("timeframe", "unknown"),
                "CAGR": m["CAGR"],
                "Sharpe": m["Sharpe"],
                "Sortino": m["Sortino"],
                "Calmar": m["Calmar"],
                "MaxDD": m["MaxDD"],
            }
        )
    return rows


def build_leaderboard(root: str | Path = "results/runs") -> pd.DataFrame:
    rows = _collect_rows(root)
    if not rows:
        return pd.DataFrame(columns=["Run", *DEFAULT_COLUMNS])
    board = pd.DataFrame(rows)
    board = board.sort_values(["Calmar", "Sharpe", "CAGR"], ascending=[False, False, False])
    return board.reset_index(drop=True)


def update_strategy_leaderboard(
    root: str | Path = "results/runs",
    out_path: str | Path = "results/strategy_leaderboard.csv",
) -> pd.DataFrame:
    board = build_leaderboard(root)
    if board.empty:
        board_export = pd.DataFrame(columns=DEFAULT_COLUMNS)
    else:
        board_export = board.loc[:, DEFAULT_COLUMNS].copy()
        board_export = board_export.sort_values(
            ["Calmar", "Sharpe", "CAGR"], ascending=[False, False, False]
        ).reset_index(drop=True)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    board_export.to_csv(out, index=False)
    return board_export


def export_leaderboard(
    root: str | Path = "results/runs", out_path: str | Path = "results/strategy_leaderboard.csv"
) -> pd.DataFrame:
    return update_strategy_leaderboard(root=root, out_path=out_path)
