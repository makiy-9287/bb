"""
Continuously monitors active trades against live prices.
When SL or TP is hit, sends an update to Telegram and removes
the trade from the active list.
"""
import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

import aiohttp

from config import (
    BINANCE_FUTURES_REST, MONITOR_INTERVAL_SEC,
    ACTIVE_TRADES_FILE
)
from telegram_sender import send_raw_message

logger = logging.getLogger(__name__)

# FIX: lock prevents race between signal_loop (add_trade) and monitor_loop
_trades_lock = asyncio.Lock()


def _load_trades_unlocked() -> List[Dict[str, Any]]:
    if os.path.exists(ACTIVE_TRADES_FILE):
        try:
            with open(ACTIVE_TRADES_FILE) as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_trades_unlocked(trades: List[Dict[str, Any]]):
    with open(ACTIVE_TRADES_FILE, "w") as f:
        json.dump(trades, f, indent=2)


async def add_trade(signal: Dict[str, Any]):
    """Register a new trade for monitoring (thread-safe)."""
    async with _trades_lock:
        trades = _load_trades_unlocked()
        signal = dict(signal)  # copy
        signal["opened_at"] = datetime.utcnow().isoformat()
        signal["status"] = "open"
        trades.append(signal)
        _save_trades_unlocked(trades)
    logger.info("Trade added to monitor: %s", signal["symbol"])


async def _get_prices(symbols: List[str]) -> Dict[str, float]:
    """Fetch last prices for a list of symbols via the ticker endpoint."""
    url = f"{BINANCE_FUTURES_REST}/fapi/v1/ticker/price"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=10) as resp:
            data = await resp.json()
    price_map = {item["symbol"]: float(item["price"]) for item in data}
    return {s: price_map.get(s) for s in symbols if price_map.get(s)}


async def monitor_loop():
    """Infinite loop: check every active trade against live price."""
    logger.info("Trade monitor started (interval=%ds)", MONITOR_INTERVAL_SEC)
    while True:
        try:
            async with _trades_lock:
                trades = _load_trades_unlocked()

            open_trades = [t for t in trades if t.get("status") == "open"]
            if not open_trades:
                await asyncio.sleep(MONITOR_INTERVAL_SEC)
                continue

            symbols = list({t["symbol"] for t in open_trades})
            prices = await _get_prices(symbols)

            remaining = []
            changed = False

            for trade in trades:
                if trade.get("status") != "open":
                    remaining.append(trade)
                    continue

                sym   = trade["symbol"]
                price = prices.get(sym)
                if price is None:
                    remaining.append(trade)
                    continue

                direction = trade["direction"].upper()
                hit = None

                if direction == "LONG":
                    if price <= trade["sl"]:
                        hit = "SL"
                    elif price >= trade["tp3"]:
                        hit = "TP3"
                    elif price >= trade["tp2"]:
                        hit = "TP2"
                    elif price >= trade["tp1"]:
                        hit = "TP1"
                else:  # SHORT
                    if price >= trade["sl"]:
                        hit = "SL"
                    elif price <= trade["tp3"]:
                        hit = "TP3"
                    elif price <= trade["tp2"]:
                        hit = "TP2"
                    elif price <= trade["tp1"]:
                        hit = "TP1"

                if hit:
                    trade["status"] = "closed"
                    trade["closed_at"] = datetime.utcnow().isoformat()
                    trade["exit_price"] = price
                    trade["exit_reason"] = hit
                    changed = True
                    emoji = "✅" if hit.startswith("TP") else "❌"
                    await send_raw_message(
                        f"{emoji} *Trade Closed* — `{sym}`\n"
                        f"Reason: *{hit}* at `{price}`\n"
                        f"Direction: {direction}"
                    )
                    logger.info("Trade closed: %s %s at %s", sym, hit, price)
                else:
                    remaining.append(trade)

            if changed:
                async with _trades_lock:
                    _save_trades_unlocked(remaining)

        except Exception as exc:
            logger.error("Monitor loop error: %s", exc)

        await asyncio.sleep(MONITOR_INTERVAL_SEC)