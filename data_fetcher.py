"""
Fetches top-volume coins + OHLCV for 1D/4H/1H/15m.
Per-timeframe candle limits applied after fetch.
"""
import asyncio
import logging
from typing import Dict, List

import aiohttp
import ccxt.async_support as ccxt
import pandas as pd

from config import (
    BINANCE_FUTURES_REST, TOP_COINS_LIMIT, BLACKLIST_SYMBOLS,
    TIMEFRAMES, CANDLE_LIMITS, OHLCV_FETCH_LIMIT,
)

logger = logging.getLogger(__name__)

_CONCURRENCY = 10
_RETRIES     = 3


async def get_top_volume_symbols(limit: int = TOP_COINS_LIMIT) -> List[str]:
    url = f"{BINANCE_FUTURES_REST}/fapi/v1/ticker/24hr"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            data = await resp.json(content_type=None)

    if not isinstance(data, list):
        # e.g. geo-block / rate-limit returns a dict like {"code":..,"msg":..}
        raise RuntimeError(f"Binance ticker error: {str(data)[:200]}")

    usdt_perps = [
        t for t in data
        if t["symbol"].endswith("USDT")
        and t["symbol"] not in BLACKLIST_SYMBOLS
    ]
    usdt_perps.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    symbols = [t["symbol"] for t in usdt_perps[:limit]]
    logger.info("Fetched top %d symbols (blacklist excluded)", len(symbols))
    return symbols


async def _fetch_one(
    exchange: ccxt.Exchange, symbol: str, timeframe: str
) -> pd.DataFrame:
    for attempt in range(_RETRIES):
        try:
            raw = await exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, limit=OHLCV_FETCH_LIMIT
            )
            df = pd.DataFrame(
                raw, columns=["ts", "open", "high", "low", "close", "volume"]
            )
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            need = CANDLE_LIMITS[timeframe]
            return df.tail(need).reset_index(drop=True)
        except Exception as exc:
            if attempt == _RETRIES - 1:
                logger.warning("OHLCV fail %s %s: %s",
                               symbol, timeframe, exc)
                return pd.DataFrame()
            await asyncio.sleep(1 + attempt)
    return pd.DataFrame()


async def fetch_all_data(
    symbols: List[str],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    sem = asyncio.Semaphore(_CONCURRENCY)
    try:
        await exchange.load_markets()   # load once, avoid 240 tasks racing to load
    except Exception as exc:
        await exchange.close()
        raise RuntimeError(f"load_markets failed: {exc}") from exc

    async def _task(sym: str, tf: str):
        async with sem:
            df = await _fetch_one(exchange, sym, tf)
            return sym, tf, df

    tasks = [_task(s, tf) for s in symbols for tf in TIMEFRAMES]
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        await exchange.close()      # always close, even on error

    bundle: Dict[str, Dict[str, pd.DataFrame]] = {}
    for r in results:
        if isinstance(r, Exception):
            logger.warning("Fetch task failed: %s", r)
            continue
        sym, tf, df = r
        if df.empty:
            continue
        bundle.setdefault(sym, {})[tf] = df

    logger.info("Fetched data for %d symbols", len(bundle))
    return bundle
