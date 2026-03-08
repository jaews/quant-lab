"""I/O utilities for strategy run folders."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REQUIRED_FILES = ("equity.csv", "trades.csv", "meta.json")
REQUIRED_EQUITY_COLUMNS = {"timestamp", "equity"}
REQUIRED_TRADES_COLUMNS = {"timestamp"}


class RunValidationError(ValueError):
    """Raised when a run folder exists but its content is incomplete or invalid."""


def load_run(run_dir: str | Path) -> dict:
    run_path = Path(run_dir)
    if not run_path.exists() or not run_path.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_path}")

    for req in REQUIRED_FILES:
        if not (run_path / req).exists():
            raise FileNotFoundError(
                f"Run '{run_path.name}' is missing required file '{req}'. "
                f"Expected files: {', '.join(REQUIRED_FILES)}."
            )

    try:
        equity = pd.read_csv(run_path / "equity.csv", parse_dates=["timestamp"])
    except Exception as exc:
        raise RunValidationError(f"Run '{run_path.name}' has invalid equity.csv format: {exc}") from exc
    try:
        trades = pd.read_csv(run_path / "trades.csv", parse_dates=["timestamp"])
    except Exception as exc:
        raise RunValidationError(f"Run '{run_path.name}' has invalid trades.csv format: {exc}") from exc
    try:
        meta = json.loads((run_path / "meta.json").read_text())
    except Exception as exc:
        raise RunValidationError(f"Run '{run_path.name}' has invalid meta.json format: {exc}") from exc

    missing_equity_cols = REQUIRED_EQUITY_COLUMNS.difference(equity.columns)
    if missing_equity_cols:
        raise RunValidationError(
            f"Run '{run_path.name}' equity.csv is missing columns: {sorted(missing_equity_cols)}"
        )
    missing_trades_cols = REQUIRED_TRADES_COLUMNS.difference(trades.columns)
    if missing_trades_cols:
        raise RunValidationError(
            f"Run '{run_path.name}' trades.csv is missing columns: {sorted(missing_trades_cols)}"
        )
    if equity.empty:
        raise RunValidationError(f"Run '{run_path.name}' equity.csv is empty.")
    if not isinstance(meta, dict):
        raise RunValidationError(f"Run '{run_path.name}' meta.json must be a JSON object.")

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
    return sorted([p for p in root_path.iterdir() if p.is_dir()])
