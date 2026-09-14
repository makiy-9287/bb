"""
Fetches top-volume coins and historical OHLCV data from Binance Futures.
Uses CCXT REST for bulk history and raw REST for ticker/volume filtering.
"""
import asyncio
import logging
from typing import Dict, List

import ccxt.async_support as ccxt
import pandas as pd
import aiohttp

from config import (
    BINANCE_FUTURES_REST, TOP_COINS_LIMIT, CANDLE_LIMIT,
    OHLCV_FETCH_LIMIT, TIMEFRAMES, DATA_DIR
)

logger = logging.getLogger(__name__)


async def get_top_volume_symbols(limit: int = TOP_COINS_LIMIT) -> List[str]:
    """Return the top `limit` USDT-margined perpetual futures symbols
    sorted by 24 h quote volume."""
    url = f"{BINANCE_FUTURES_REST}/fapi/v1/ticker/24hr"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=15) as resp:
            data = await resp.json()

    usdt_perps = [
        t for t in data
        if t["symbol"].endswith("USDT")
        and t["symbol"] not in ("USDCUSDT",)
    ]
    usdt_perps.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    symbols = [t["symbol"] for t in usdt_perps[:limit]]
    logger.info("Fetched top %d symbols by volume", len(symbols))
    return symbols


async def fetch_ohlcv(
    exchange: ccxt.Exchange, symbol: str, timeframe: str
) -> pd.DataFrame:
    """Fetch OHLCV as a DataFrame. Uses the optimised 499-candle limit."""
    try:
        raw = await exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, limit=OHLCV_FETCH_LIMIT
        )
        df = pd.DataFrame(
            raw, columns=["ts", "open", "high", "low", "close", "volume"]
        )
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df = df.tail(CANDLE_LIMIT).reset_index(drop=True)
        return df
    except Exception as exc:
        logger.warning("OHLCV fetch failed %s %s: %s", symbol, timeframe, exc)
        return pd.DataFrame()


async def fetch_all_data(symbols: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Fetch OHLCV for every symbol × timeframe."""
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })

    async def _one(sym: str, tf: str):
        df = await fetch_ohlcv(exchange, sym, tf)
        return sym, tf, df

    tasks = [_one(s, tf) for s in symbols for tf in TIMEFRAMES]
    results = await asyncio.gather(*tasks)

    bundle: Dict[str, Dict[str, pd.DataFrame]] = {}
    for sym, tf, df in results:
        if df.empty:
            continue
        bundle.setdefault(sym, {})[tf] = df

    await exchange.close()
    logger.info("Fetched data for %d symbols", len(bundle))
    return bundle