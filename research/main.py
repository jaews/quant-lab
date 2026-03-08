"""Unified CLI for the continuous quant research lab."""

from __future__ import annotations

import argparse
import logging

from analysis.compare import update_strategy_leaderboard
from pipeline.run_backtests import run_backtests
from pipeline.update_data import update_market_data


def run_pipeline() -> None:
    update_market_data()
    run_backtests()
    update_strategy_leaderboard()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Continuous quant research lab CLI")
    parser.add_argument(
        "command",
        choices=["update-data", "run-backtests", "update-leaderboard", "run-pipeline"],
        help="Pipeline command to execute",
    )
    args = parser.parse_args()

    if args.command == "update-data":
        update_market_data()
    elif args.command == "run-backtests":
        run_backtests()
    elif args.command == "update-leaderboard":
        update_strategy_leaderboard()
    elif args.command == "run-pipeline":
        run_pipeline()


if __name__ == "__main__":
    main()
