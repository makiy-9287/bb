"""
Binance Futures data fetcher — concurrency-limited, retry-safe.
"""
import asyncio
import logging
from typing import Dict, List

import aiohttp
import ccxt.async_support as ccxt
import pandas as pd

from config import (
    BINANCE_FUTURES_REST, TOP_COINS_LIMIT,
    MAX_CANDLES, TIMEFRAMES,
)

logger = logging.getLogger(__name__)

_CONCURRENCY = 10
_RETRIES     = 3


async def get_top_volume_symbols(limit: int = TOP_COINS_LIMIT) -> List[str]:
    """Top USDT perpetuals by volume, filtered for real liquidity."""
    url = f"{BINANCE_FUTURES_REST}/fapi/v1/ticker/24hr"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=15) as resp:
                if resp.status != 200:
                    logger.error("Binance HTTP %d", resp.status)
                    return []
                data = await resp.json()
    except Exception as exc:
        logger.error("Binance API failed: %s", exc)
        return []

    if not isinstance(data, list):
        logger.error("Binance returned non-list: %s", str(data)[:200])
        return []

    MIN_VOLUME_USD  = 50_000_000
    MIN_TRADE_COUNT = 200_000
    TOKENIZED = {
        "AAPLUSDT", "MSFTUSDT", "GOOGLUSDT", "AMZNUSDT", "METAUSDT",
        "NVDAUSDT", "TSLAUSDT", "NFLXUSDT", "AMDUSDT", "COINUSDT",
        "MSTRUSDT", "SPCXUSDT", "PLTRUSDT", "CRCLUSDT", "HOODUSDT",
    }
    BLACKLIST = {"USDCUSDT"}

    filtered = []
    for t in data:
        if not isinstance(t, dict):
            continue
        sym = t.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        if sym in BLACKLIST or sym in TOKENIZED:
            continue
        try:
            vol = float(t.get("quoteVolume", 0))
            cnt = int(t.get("count", 0))
        except (ValueError, TypeError):
            continue
        if vol < MIN_VOLUME_USD or cnt < MIN_TRADE_COUNT:
            continue
        filtered.append(t)

    if not filtered:
        logger.warning("No symbols passed filters")
        return []

    filtered.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    symbols = [t["symbol"] for t in filtered[:limit]]
    logger.info("Top %d symbols — sample: %s", len(symbols), symbols[:5])
    return symbols


async def fetch_ohlcv(
    exchange: ccxt.Exchange, symbol: str, timeframe: str,
) -> pd.DataFrame:
    for attempt in range(_RETRIES):
        try:
            raw = await exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, limit=MAX_CANDLES
            )
            df = pd.DataFrame(
                raw, columns=["ts", "open", "high", "low", "close", "volume"]
            )
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            return df
        except Exception as exc:
            if attempt == _RETRIES - 1:
                logger.warning("OHLCV %s %s: %s", symbol, timeframe, exc)
                return pd.DataFrame()
            await asyncio.sleep(1 + attempt)
    return pd.DataFrame()


async def fetch_all_data(
    symbols: List[str],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Fetch OHLCV for every symbol × timeframe (max 10 concurrent)."""
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
