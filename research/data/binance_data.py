"""Binance OHLCV ingestion with incremental local CSV caching."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

SUPPORTED_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT")
SUPPORTED_TIMEFRAMES = ("1h", "4h", "1d")
TIMEFRAME_MS = {"1h": 60 * 60 * 1000, "4h": 4 * 60 * 60 * 1000, "1d": 24 * 60 * 60 * 1000}
DEFAULT_CACHE_DIR = Path("data/cache")


def _to_exchange_symbol(symbol: str) -> str:
    return f"{symbol[:-4]}/USDT"


def _cache_path(symbol: str, timeframe: str, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> Path:
    return Path(cache_dir) / f"{symbol}_{timeframe}.csv"


def _read_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Invalid cache file {path}: missing columns {sorted(missing)}")
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def _save_cache(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("timestamp").drop_duplicates("timestamp").to_csv(path, index=False)


def _retry_fetch(exchange: object, symbol: str, timeframe: str, since: int | None, limit: int, retries: int = 3) -> list[list]:
    wait = 1.0
    for attempt in range(1, retries + 1):
        try:
            return exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        except Exception as exc:
            if attempt == retries:
                raise
            LOGGER.warning(
                "Binance fetch failed (%s %s, attempt %s/%s): %s. Retrying in %.1fs",
                symbol,
                timeframe,
                attempt,
                retries,
                exc,
                wait,
            )
            time.sleep(wait)
            wait *= 2
    return []


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    limit: int = 1000,
    max_candles: int = 5000,
) -> pd.DataFrame:
    if symbol not in SUPPORTED_SYMBOLS:
        raise ValueError(f"Unsupported symbol '{symbol}'. Allowed: {', '.join(SUPPORTED_SYMBOLS)}")
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Allowed: {', '.join(SUPPORTED_TIMEFRAMES)}")

    try:
        import ccxt
    except ImportError as exc:
        raise ImportError("ccxt is required for Binance data ingestion. Install with `pip install -r requirements.txt`.") from exc

    cache_path = _cache_path(symbol, timeframe, cache_dir)
    cached = _read_cache(cache_path)

    since: int | None = None
    if not cached.empty:
        last_ts = int(cached["timestamp"].iloc[-1].timestamp() * 1000)
        since = last_ts + TIMEFRAME_MS[timeframe]

    exchange = ccxt.binance({"enableRateLimit": True})
    exchange_symbol = _to_exchange_symbol(symbol)

    all_new: list[pd.DataFrame] = []
    fetched = 0
    while fetched < max_candles:
        remaining = max_candles - fetched
        batch_limit = min(limit, remaining)
        candles = _retry_fetch(exchange, exchange_symbol, timeframe, since, batch_limit)
        if not candles:
            break

        batch = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        batch["timestamp"] = pd.to_datetime(batch["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
        batch = batch.sort_values("timestamp").drop_duplicates("timestamp")
        if since is not None:
            batch = batch[batch["timestamp"] > pd.to_datetime(since, unit="ms")]
        if batch.empty:
            break
        all_new.append(batch)

        fetched += len(batch)
        since = int(batch["timestamp"].iloc[-1].timestamp() * 1000) + TIMEFRAME_MS[timeframe]
        if len(candles) < batch_limit:
            break

    if all_new:
        new_df = pd.concat(all_new, ignore_index=True)
        merged = new_df if cached.empty else pd.concat([cached, new_df], ignore_index=True)
        merged = merged.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        _save_cache(cache_path, merged)
        LOGGER.info("Updated %s (%s): +%s candles", symbol, timeframe, len(new_df))
        return merged

    if not cached.empty:
        LOGGER.info("No new candles for %s (%s)", symbol, timeframe)
    else:
        LOGGER.warning("No candles fetched for %s (%s)", symbol, timeframe)
    return cached
