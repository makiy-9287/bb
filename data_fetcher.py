"""
Binance Futures data fetcher — concurrency-safe with retries.
"""
import asyncio
import logging
from typing import Dict, List

import aiohttp
import ccxt.async_support as ccxt
import pandas as pd

from config import (
    BINANCE_FUTURES_REST, TOP_COINS_LIMIT, CANDLE_LIMIT,
    OHLCV_FETCH_LIMIT, TIMEFRAMES,
)

logger = logging.getLogger(__name__)

_CONCURRENCY = 10
_RETRIES     = 3


async def get_top_volume_symbols(limit: int = TOP_COINS_LIMIT) -> List[str]:
    url = f"{BINANCE_FUTURES_REST}/fapi/v1/ticker/24hr"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=15) as resp:
            data = await resp.json()
    usdt = [t for t in data
            if t["symbol"].endswith("USDT") and t["symbol"] != "USDCUSDT"]
    usdt.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    symbols = [t["symbol"] for t in usdt[:limit]]
    logger.info("Top %d symbols by volume", len(symbols))
    return symbols


async def fetch_ohlcv(
    exchange: ccxt.Exchange, symbol: str, timeframe: str,
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
            return df.tail(CANDLE_LIMIT).reset_index(drop=True)
        except Exception as exc:
            if attempt == _RETRIES - 1:
                logger.warning("OHLCV %s %s failed: %s",
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

    async def _one(sym, tf):
        async with sem:
            df = await fetch_ohlcv(exchange, sym, tf)
            return sym, tf, df

    tasks = [_one(s, tf) for s in symbols for tf in TIMEFRAMES]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    bundle: Dict[str, Dict[str, pd.DataFrame]] = {}
    for r in results:
        if isinstance(r, Exception):
            logger.warning("Fetch task failed: %s", r)
            continue
        sym, tf, df = r
        if df.empty:
            continue
        bundle.setdefault(sym, {})[tf] = df

    await exchange.close()
    logger.info("Fetched %d symbols", len(bundle))
    return bundle
