"""Daily market data update pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from data.binance_data import (
    DEFAULT_CACHE_DIR,
    SUPPORTED_SYMBOLS,
    SUPPORTED_TIMEFRAMES,
    fetch_ohlcv,
)

LOGGER = logging.getLogger(__name__)


def update_market_data(
    symbols: tuple[str, ...] = SUPPORTED_SYMBOLS,
    timeframes: tuple[str, ...] = SUPPORTED_TIMEFRAMES,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    retries: int = 1,
) -> dict:
    summary = {"updated": 0, "failed": 0}
    for symbol in symbols:
        for timeframe in timeframes:
            success = False
            wait = 1.0
            for attempt in range(1, retries + 1):
                try:
                    fetch_ohlcv(symbol=symbol, timeframe=timeframe, cache_dir=cache_dir)
                    summary["updated"] += 1
                    success = True
                    break
                except Exception as exc:
                    LOGGER.error(
                        "Update failed for %s %s (attempt %s/%s): %s",
                        symbol,
                        timeframe,
                        attempt,
                        retries,
                        exc,
                    )
                    if attempt < retries:
                        time.sleep(wait)
                        wait *= 2
            if not success:
                summary["failed"] += 1
    LOGGER.info("Data update complete: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    update_market_data()
