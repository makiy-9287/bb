"""
Binance USDT-M perpetual data: universe (exchange metadata + liquidity filters), server time,
live prices, and cleaned CLOSED-candle OHLCV (see ohlcv.py).
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import pandas as pd

from config import (BINANCE_FUTURES_REST, TOP_COINS_LIMIT, BLACKLIST_SYMBOLS, TIMEFRAMES, CANDLE_LIMITS,
                    MIN_QUOTE_VOLUME_USDT, MAX_SPREAD_PCT, MAX_DATA_STALE_BARS)
from ohlcv import clean_ohlcv

logger = logging.getLogger(__name__)
_CONCURRENCY = 10
_RETRIES = 3
_TO = aiohttp.ClientTimeout(total=15)


async def _get_json(session: aiohttp.ClientSession, path: str, **params) -> Any:
    last: Optional[Exception] = None
    for attempt in range(_RETRIES):
        try:
            async with session.get(f"{BINANCE_FUTURES_REST}{path}", params=params or None, timeout=_TO) as r:
                data = await r.json(content_type=None)
                if r.status == 200:
                    return data
                last = RuntimeError(f"HTTP {r.status}: {str(data)[:150]}")
        except Exception as exc:
            last = exc
        await asyncio.sleep(1 + attempt)
    raise RuntimeError(f"{path} failed: {last}")


async def get_server_time_ms() -> int:
    async with aiohttp.ClientSession() as s:
        return int((await _get_json(s, "/fapi/v1/time"))["serverTime"])


async def get_last_prices(symbols: List[str]) -> Dict[str, float]:
    async with aiohttp.ClientSession() as s:
        data = await _get_json(s, "/fapi/v1/ticker/price")
    pm = {i["symbol"]: float(i["price"]) for i in data if "symbol" in i}
    return {x: pm[x] for x in symbols if x in pm and pm[x] > 0}


def filter_universe(info: Dict[str, Any], tickers: List[Dict[str, Any]], books: List[Dict[str, Any]],
                    limit: int, exclude: Optional[set] = None) -> List[Dict[str, Any]]:
    """Pure function (unit-testable): active USDT PERPETUAL, liquid, tight spread, not blacklisted."""
    ok = {s["symbol"] for s in info.get("symbols", [])
          if s.get("contractType") == "PERPETUAL" and s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT"}
    spread: Dict[str, float] = {}
    for b in books:
        try:
            bid, ask = float(b["bidPrice"]), float(b["askPrice"])
            if bid > 0 and ask >= bid:
                spread[b["symbol"]] = (ask - bid) / ((ask + bid) / 2) * 100
        except (KeyError, ValueError, TypeError):
            continue
    out = []
    for t in tickers:
        sym = t.get("symbol")
        if sym not in ok or sym in BLACKLIST_SYMBOLS or (exclude and sym in exclude):
            continue
        try:
            qv, px = float(t["quoteVolume"]), float(t["lastPrice"])
        except (KeyError, ValueError, TypeError):
            continue
        sp = spread.get(sym)
        if qv < MIN_QUOTE_VOLUME_USDT or px <= 0 or sp is None or sp > MAX_SPREAD_PCT:
            continue
        out.append({"symbol": sym, "quote_volume": qv, "price": px, "spread_pct": sp})
    out.sort(key=lambda x: x["quote_volume"], reverse=True)
    return out[:limit]


async def get_universe(limit: int = TOP_COINS_LIMIT, exclude: Optional[set] = None) -> List[Dict[str, Any]]:
    async with aiohttp.ClientSession() as s:
        info, tick, books = await asyncio.gather(
            _get_json(s, "/fapi/v1/exchangeInfo"), _get_json(s, "/fapi/v1/ticker/24hr"),
            _get_json(s, "/fapi/v1/ticker/bookTicker"))
    if not isinstance(tick, list) or not isinstance(books, list) or not isinstance(info, dict):
        raise RuntimeError("unexpected Binance response shape")
    uni = filter_universe(info, tick, books, limit, exclude)
    logger.info("Universe: %d liquid active USDT perpetuals (min vol %.0fM, max spread %.3f%%)",
                len(uni), MIN_QUOTE_VOLUME_USDT / 1e6, MAX_SPREAD_PCT)
    return uni


async def get_top_volume_symbols(limit: int = TOP_COINS_LIMIT) -> List[str]:   # backward compatible
    return [u["symbol"] for u in await get_universe(limit)]


async def _fetch_raw(ex, symbol: str, tf: str, limit: int) -> List[List[float]]:
    for attempt in range(_RETRIES):
        try:
            return await ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
        except Exception as exc:
            if attempt == _RETRIES - 1:
                logger.warning("OHLCV fail %s %s: %s", symbol, tf, exc)
                return []
            await asyncio.sleep(1 + attempt)
    return []


async def fetch_bundle(symbols: List[str], tfs: List[str], now_ms: int
                       ) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, Dict[str, Any]]]]:
    """Returns (frames[sym][tf] = CLOSED, validated candles; meta[sym][tf]). Bad data is excluded, never patched."""
    import ccxt.async_support as ccxt      # lazy: only needed for live fetching
    ex = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})
    sem = asyncio.Semaphore(_CONCURRENCY)
    try:
        await ex.load_markets()
    except Exception as exc:
        await ex.close()
        raise RuntimeError(f"load_markets failed: {exc}") from exc

    async def task(sym: str, tf: str):
        async with sem:
            raw = await _fetch_raw(ex, sym, tf, CANDLE_LIMITS.get(tf, 300) + 1)   # +1: forming candle gets dropped
            return sym, tf, raw
    try:
        results = await asyncio.gather(*[task(s, tf) for s in symbols for tf in tfs], return_exceptions=True)
    finally:
        await ex.close()

    frames: Dict[str, Dict[str, pd.DataFrame]] = {}
    metas: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in results:
        if isinstance(r, Exception):
            logger.warning("Fetch task failed: %s", r)
            continue
        sym, tf, raw = r
        df, meta = clean_ohlcv(raw, tf, now_ms, MAX_DATA_STALE_BARS)
        metas.setdefault(sym, {})[tf] = meta
        if meta["ok"]:
            frames.setdefault(sym, {})[tf] = df.tail(CANDLE_LIMITS.get(tf, 300)).reset_index(drop=True)
        else:
            logger.info("Data rejected %s %s: %s", sym, tf, meta["reason"])
    logger.info("Fetched valid closed-candle data for %d/%d symbols", len(frames), len(symbols))
    return frames, metas


async def fetch_all_data(symbols: List[str]):            # backward-compatible name
    now = await get_server_time_ms()
    frames, _ = await fetch_bundle(symbols, TIMEFRAMES, now)
    return frames
