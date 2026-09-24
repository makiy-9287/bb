"""
Continuous trade monitor with race-safe writes + PnL tracking.
"""
import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

from config import (
    BINANCE_FUTURES_REST, MONITOR_INTERVAL_SEC,
    ACTIVE_TRADES_FILE, CLOSED_TRADES_FILE,
)
from telegram_sender import send_raw_message

logger = logging.getLogger(__name__)

_trades_lock: Optional[asyncio.Lock] = None


def _get_lock() -> asyncio.Lock:
    global _trades_lock
    if _trades_lock is None:
        _trades_lock = asyncio.Lock()
    return _trades_lock


def _load_json(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return []


def _save_json(path: str, data: List[Dict[str, Any]]):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def get_active_trades() -> List[Dict[str, Any]]:
    return [t for t in _load_json(ACTIVE_TRADES_FILE)
            if t.get("status") == "open"]


def get_closed_trades() -> List[Dict[str, Any]]:
    return _load_json(CLOSED_TRADES_FILE)


def calc_pnl(trade: Dict[str, Any], current_price: float) -> Dict[str, float]:
    entry = float(trade["entry"])
    if entry == 0:
        return {"pnl_pct": 0.0, "direction": trade["direction"]}
    if trade["direction"].upper() == "LONG":
        pct = (current_price - entry) / entry * 100
    else:
        pct = (entry - current_price) / entry * 100
    return {"pnl_pct": pct, "direction": trade["direction"]}


async def get_prices(symbols: List[str]) -> Dict[str, float]:
    if not symbols:
        return {}
    url = f"{BINANCE_FUTURES_REST}/fapi/v1/ticker/price"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=10) as resp:
            data = await resp.json()
    pm = {i["symbol"]: float(i["price"]) for i in data}
    return {s: pm[s] for s in symbols if s in pm}


def _trade_key(t: Dict[str, Any]) -> str:
    return f"{t.get('symbol','?')}|{t.get('opened_at','')}"


def _append_closed(trade: Dict[str, Any]):
    closed = _load_json(CLOSED_TRADES_FILE)
    closed.append(trade)
    _save_json(CLOSED_TRADES_FILE, closed)


async def add_trade(signal: Dict[str, Any]):
    async with _get_lock():
        trades = _load_json(ACTIVE_TRADES_FILE)
        t = dict(signal)
        t["opened_at"] = datetime.utcnow().isoformat()
        t["status"] = "open"
        trades.append(t)
        _save_json(ACTIVE_TRADES_FILE, trades)
    logger.info("Trade added to monitor: %s", signal["symbol"])


async def monitor_loop():
    logger.info("Trade monitor started (interval=%ds)", MONITOR_INTERVAL_SEC)

    while True:
        try:
            trades = _load_json(ACTIVE_TRADES_FILE)
            open_trades = [t for t in trades if t.get("status") == "open"]

            if not open_trades:
                await asyncio.sleep(MONITOR_INTERVAL_SEC)
                continue

            symbols = list({t["symbol"] for t in open_trades})
            try:
                prices = await get_prices(symbols)
            except Exception as exc:
                logger.warning("Price fetch failed: %s", exc)
                await asyncio.sleep(MONITOR_INTERVAL_SEC)
                continue

            # Determine which trades to close
            to_close: Dict[str, tuple] = {}
            for trade in open_trades:
                sym = trade["symbol"]
                price = prices.get(sym)
                if price is None:
                    continue
                d = trade["direction"].upper()
                hit = None
                if d == "LONG":
                    if price <= float(trade["sl"]):    hit = "SL"
                    elif price >= float(trade["tp3"]): hit = "TP3"
                    elif price >= float(trade["tp2"]): hit = "TP2"
                    elif price >= float(trade["tp1"]): hit = "TP1"
                else:
                    if price >= float(trade["sl"]):    hit = "SL"
                    elif price <= float(trade["tp3"]): hit = "TP3"
                    elif price <= float(trade["tp2"]): hit = "TP2"
                    elif price <= float(trade["tp1"]): hit = "TP1"
                if hit:
                    to_close[_trade_key(trade)] = (price, hit)

            if not to_close:
                await asyncio.sleep(MONITOR_INTERVAL_SEC)
                continue

            # Atomic update under lock (fixes race)
            notifications = []
            async with _get_lock():
                current = _load_json(ACTIVE_TRADES_FILE)   # RE-READ inside lock
                still_open: List[Dict[str, Any]] = []
                for t in current:
                    k = _trade_key(t)
                    if t.get("status") == "open" and k in to_close:
                        price, hit = to_close[k]
                        t["status"] = "closed"
                        t["closed_at"] = datetime.utcnow().isoformat()
                        t["exit_price"] = price
                        t["exit_reason"] = hit
                        pnl = calc_pnl(t, price)
                        t["pnl_pct"] = round(pnl["pnl_pct"], 4)
                        _append_closed(t)
                        notifications.append(
                            (t["symbol"], hit, price, t["pnl_pct"])
                        )
                    else:
                        still_open.append(t)
                _save_json(ACTIVE_TRADES_FILE, still_open)

            for sym, hit, price, pnl_pct in notifications:
                emoji = "✅" if hit.startswith("TP") else "❌"
                await send_raw_message(
                    f"{emoji} *Trade Closed* — `{sym}`\n"
                    f"Reason: *{hit}* at `{price}`\n"
                    f"PnL: `{pnl_pct:+.2f}%`"
                )
                logger.info("Trade closed: %s %s at %s (%.2f%%)",
                            sym, hit, price, pnl_pct)

        except Exception as exc:
            logger.error("Monitor loop error: %s", exc, exc_info=True)

        await asyncio.sleep(MONITOR_INTERVAL_SEC)
