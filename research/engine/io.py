"""I/O utilities for strategy run folders."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REQUIRED_FILES = ("equity.csv", "trades.csv", "meta.json")


def load_run(run_dir: str | Path) -> dict:
    run_path = Path(run_dir)
    for req in REQUIRED_FILES:
        if not (run_path / req).exists():
            raise FileNotFoundError(f"Missing {req} in {run_path}")
    equity = pd.read_csv(run_path / "equity.csv", parse_dates=["timestamp"])
    trades = pd.read_csv(run_path / "trades.csv", parse_dates=["timestamp"])
    meta = json.loads((run_path / "meta.json").read_text())
    return {"name": run_path.name, "equity": equity, "trades": trades, "meta": meta}


def save_run(run_dir: str | Path, equity: pd.DataFrame, trades: pd.DataFrame, meta: dict) -> None:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    equity.to_csv(run_path / "equity.csv", index=False)
    trades.to_csv(run_path / "trades.csv", index=False)
    (run_path / "meta.json").write_text(json.dumps(meta, indent=2))


def list_runs(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted([p for p in root_path.iterdir() if p.is_dir() and (p / "equity.csv").exists()])
