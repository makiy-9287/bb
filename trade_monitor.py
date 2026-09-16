"""
Continuous trade monitor — checks SL/TP against live prices every
MONITOR_INTERVAL_SEC. Sends Telegram updates on close.
Tracks closed-trade history for /pnl command.
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

# Lazy lock — created on first use (safe across event loops)
_trades_lock: Optional[asyncio.Lock] = None


def _get_lock() -> asyncio.Lock:
    global _trades_lock
    if _trades_lock is None:
        _trades_lock = asyncio.Lock()
    return _trades_lock


# ── File I/O ─────────────────────────────────────────────────────
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


# ── Public accessors (used by telegram_bot.py) ───────────────────
def get_active_trades() -> List[Dict[str, Any]]:
    return [t for t in _load_json(ACTIVE_TRADES_FILE)
            if t.get("status") == "open"]


def get_closed_trades() -> List[Dict[str, Any]]:
    return _load_json(CLOSED_TRADES_FILE)


def calc_pnl(trade: Dict[str, Any], current_price: float) -> Dict[str, float]:
    """Return {'pnl_pct': float, 'direction': str}."""
    entry = float(trade["entry"])
    if entry == 0:
        return {"pnl_pct": 0.0, "direction": trade["direction"]}
    if trade["direction"].upper() == "LONG":
        pct = (current_price - entry) / entry * 100
    else:
        pct = (entry - current_price) / entry * 100
    return {"pnl_pct": pct, "direction": trade["direction"]}


# ── Binance price fetch ──────────────────────────────────────────
async def get_prices(symbols: List[str]) -> Dict[str, float]:
    """Fetch last price for each symbol via the ticker endpoint."""
    url = f"{BINANCE_FUTURES_REST}/fapi/v1/ticker/price"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=10) as resp:
            data = await resp.json()
    price_map = {item["symbol"]: float(item["price"]) for item in data}
    return {s: price_map[s] for s in symbols if s in price_map}


# ── Trade lifecycle ──────────────────────────────────────────────
async def add_trade(signal: Dict[str, Any]):
    """Register a new trade for monitoring."""
    async with _get_lock():
        trades = _load_json(ACTIVE_TRADES_FILE)
        t = dict(signal)
        t["opened_at"] = datetime.utcnow().isoformat()
        t["status"] = "open"
        trades.append(t)
        _save_json(ACTIVE_TRADES_FILE, trades)
    logger.info("Trade added to monitor: %s", signal["symbol"])


async def _close_trade(trade: Dict[str, Any], exit_price: float, reason: str):
    """Move trade from active to closed with PnL snapshot."""
    trade["status"] = "closed"
    trade["closed_at"] = datetime.utcnow().isoformat()
    trade["exit_price"] = exit_price
    trade["exit_reason"] = reason
    pnl = calc_pnl(trade, exit_price)
    trade["pnl_pct"] = round(pnl["pnl_pct"], 4)

    async with _get_lock():
        closed = _load_json(CLOSED_TRADES_FILE)
        closed.append(trade)
        _save_json(CLOSED_TRADES_FILE, closed)


# ── Main loop ────────────────────────────────────────────────────
async def monitor_loop():
    """Check active trades against live price every MONITOR_INTERVAL_SEC."""
    logger.info("Trade monitor started (interval=%ds)", MONITOR_INTERVAL_SEC)

    while True:
        try:
            trades = _load_json(ACTIVE_TRADES_FILE)
            open_trades = [t for t in trades if t.get("status") == "open"]

            if not open_trades:
                await asyncio.sleep(MONITOR_INTERVAL_SEC)
                continue

            symbols = list({t["symbol"] for t in open_trades})
            prices = await get_prices(symbols)

            remaining: List[Dict[str, Any]] = []
            changed = False

            for trade in trades:
                if trade.get("status") != "open":
                    remaining.append(trade)
                    continue

                sym = trade["symbol"]
                price = prices.get(sym)
                if price is None:
                    remaining.append(trade)
                    continue

                direction = trade["direction"].upper()
                hit = None

                if direction == "LONG":
                    if price <= float(trade["sl"]):       hit = "SL"
                    elif price >= float(trade["tp3"]):    hit = "TP3"
                    elif price >= float(trade["tp2"]):    hit = "TP2"
                    elif price >= float(trade["tp1"]):    hit = "TP1"
                else:
                    if price >= float(trade["sl"]):       hit = "SL"
                    elif price <= float(trade["tp3"]):    hit = "TP3"
                    elif price <= float(trade["tp2"]):    hit = "TP2"
                    elif price <= float(trade["tp1"]):    hit = "TP1"

                if hit:
                    await _close_trade(trade, price, hit)
                    changed = True
                    emoji = "✅" if hit.startswith("TP") else "❌"
                    pnl_pct = trade.get("pnl_pct", 0.0)
                    await send_raw_message(
                        f"{emoji} *Trade Closed* — `{sym}`\n"
                        f"Reason: *{hit}* at `{price}`\n"
                        f"Direction: {direction}\n"
                        f"PnL: `{pnl_pct:+.2f}%`"
                    )
                    logger.info("Trade closed: %s %s at %s (%.2f%%)",
                                sym, hit, price, pnl_pct)
                else:
                    remaining.append(trade)

            if changed:
                async with _get_lock():
                    _save_json(ACTIVE_TRADES_FILE, remaining)

        except Exception as exc:
            logger.error("Monitor loop error: %s", exc, exc_info=True)

        await asyncio.sleep(MONITOR_INTERVAL_SEC)
