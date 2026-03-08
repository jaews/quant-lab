"""Continuous scheduler for daily quant research refresh."""

from __future__ import annotations

import logging
import time

import schedule

from analysis.compare import update_strategy_leaderboard
from pipeline.run_backtests import run_backtests
from pipeline.update_data import update_market_data

LOGGER = logging.getLogger(__name__)


def run_full_pipeline() -> None:
    LOGGER.info("Starting scheduled quant pipeline run")
    update_market_data()
    run_backtests()
    update_strategy_leaderboard()
    LOGGER.info("Completed scheduled quant pipeline run")


def start_scheduler() -> None:
    schedule.every().day.at("02:00").do(run_full_pipeline)
    LOGGER.info("Scheduler started. Daily run configured at 02:00.")
    while True:
        schedule.run_pending()
        time.sleep(20)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    start_scheduler()
