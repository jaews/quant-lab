"""Strategy comparison and leaderboard generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.metrics import compute_metrics
from engine.io import list_runs, load_run


def build_leaderboard(root: str | Path = "results/runs") -> pd.DataFrame:
    rows = []
    for run_path in list_runs(root):
        run = load_run(run_path)
        m = compute_metrics(run["equity"]["equity"], run["trades"])
        rows.append(
            {
                "Run": run["name"],
                "Strategy": run["meta"].get("strategy_name", "unknown"),
                "CAGR": m["CAGR"],
                "Sharpe": m["Sharpe"],
                "Sortino": m["Sortino"],
                "Calmar": m["Calmar"],
                "MaxDD": m["MaxDD"],
            }
        )
    if not rows:
        return pd.DataFrame(columns=["Run", "Strategy", "CAGR", "Sharpe", "Sortino", "Calmar", "MaxDD"])
    board = pd.DataFrame(rows)
    board = board.sort_values(["Calmar", "Sharpe", "CAGR", "MaxDD"], ascending=[False, False, False, False])
    return board.reset_index(drop=True)


def export_leaderboard(root: str | Path = "results/runs", out_path: str | Path = "results/strategy_leaderboard.csv") -> pd.DataFrame:
    board = build_leaderboard(root)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(out, index=False)
    return board
