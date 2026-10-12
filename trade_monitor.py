"""
Continuous trade monitor with TP1/TP2 partial tracking.

Lifecycle:
  • TP1 hit → notify, move SL to breakeven (trade stays open)
  • TP2 hit → notify, move SL to TP1     (trade stays open)
  • TP3 hit → close (final PnL)
  • SL hit  → close (with context: raw / breakeven / trailing)
"""
import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from config import (
    BINANCE_FUTURES_REST, MONITOR_INTERVAL_SEC,
    ACTIVE_TRADES_FILE, CLOSED_TRADES_FILE,
)
from telegram_sender import send_raw_message

logger = logging.getLogger(__name__)
_lock: Optional[asyncio.Lock] = None


def _get_lock() -> asyncio.Lock:
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    return _lock


# ── File I/O ─────────────────────────────────────────────────────
def _load_json(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as exc:
        logger.warning("Read %s failed: %s", path, exc)
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


# ── Trade lifecycle ──────────────────────────────────────────────
async def add_trade(signal: Dict[str, Any]):
    async with _get_lock():
        trades = _load_json(ACTIVE_TRADES_FILE)
        t = dict(signal)
        t["opened_at"] = datetime.utcnow().isoformat()
        t["status"]    = "open"
        t["tp1_hit"]   = False
        t["tp2_hit"]   = False
        t["tp3_hit"]   = False
        trades.append(t)
        _save_json(ACTIVE_TRADES_FILE, trades)
    logger.info("Trade added: %s", signal["symbol"])


# ── Exit detection ───────────────────────────────────────────────
def _detect_action(trade: Dict[str, Any], price: float) -> Optional[str]:
    """
    Return SL / TP1 / TP2 / TP3 or None.
    Order matters: SL is checked first (protective), then TP3, TP2, TP1.
    """
    d = trade["direction"].upper()
    sl  = float(trade["sl"])
    tp1 = float(trade["tp1"])
    tp2 = float(trade["tp2"])
    tp3 = float(trade["tp3"])
    tp1_hit = bool(trade.get("tp1_hit"))
    tp2_hit = bool(trade.get("tp2_hit"))

    if d == "LONG":
        if price <= sl:
            return "SL"
        if not tp2_hit and price >= tp3:
            return "TP3"
        if not tp2_hit and price >= tp2:
            return "TP2"
        if not tp1_hit and price >= tp1:
            return "TP1"
    else:  # SHORT
        if price >= sl:
            return "SL"
        if not tp2_hit and price <= tp3:
            return "TP3"
        if not tp2_hit and price <= tp2:
            return "TP2"
        if not tp1_hit and price <= tp1:
            return "TP1"
    return None


def _cascade_tp_flags(t: Dict[str, Any], action: str) -> None:
    """
    BUG FIX: If price gaps past multiple TP levels in one cycle,
    mark the intermediate TPs as hit so the record is consistent.
    """
    if action == "TP2":
        t["tp1_hit"] = True
        t["tp2_hit"] = True
    elif action == "TP3":
        t["tp1_hit"] = True
        t["tp2_hit"] = True
        t["tp3_hit"] = True


def _sl_context(t: Dict[str, Any]) -> str:
    """Human-readable suffix describing how SL was positioned."""
    if t.get("tp2_hit"):
        return " (trailed to TP1)"
    if t.get("tp1_hit"):
        return " (breakeven)"
    return ""


# ── Main loop ────────────────────────────────────────────────────
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

            notifications: List[Tuple[Dict[str, Any], str, float]] = []

            async with _get_lock():
                current = _load_json(ACTIVE_TRADES_FILE)
                updated: List[Dict[str, Any]] = []

                for t in current:
                    if t.get("status") != "open":
                        updated.append(t)
                        continue

                    sym = t["symbol"]
                    price = prices.get(sym)
                    if price is None:
                        updated.append(t)
                        continue

                    action = _detect_action(t, price)
                    if action is None:
                        updated.append(t)
                        continue

                    if action == "TP1":
                        t["tp1_hit"] = True
                        t["sl"] = t["entry"]          # move to breakeven
                        updated.append(t)
                        notifications.append((t, "TP1", price))

                    elif action == "TP2":
                        _cascade_tp_flags(t, "TP2")
                        t["sl"] = t["tp1"]            # trail to TP1
                        updated.append(t)
                        notifications.append((t, "TP2", price))

                    else:  # SL or TP3 → close
                        if action == "TP3":
                            _cascade_tp_flags(t, "TP3")
                        t["status"]      = "closed"
                        t["closed_at"]   = datetime.utcnow().isoformat()
                        t["exit_price"]  = price
                        t["exit_reason"] = action
                        pnl = calc_pnl(t, price)
                        t["pnl_pct"] = round(pnl["pnl_pct"], 4)
                        _append_closed(t)
                        notifications.append((t, action, price))

                _save_json(ACTIVE_TRADES_FILE, updated)

            # ── Send notifications outside lock ─────────────────
            for t, action, price in notifications:
                sym = t["symbol"]
                d = t["direction"].upper()

                if action in ("TP1", "TP2"):
                    pnl_now = calc_pnl(t, price)["pnl_pct"]
                    extra = ("SL → breakeven" if action == "TP1"
                             else "SL → TP1")
                    await send_raw_message(
                        f"🎯 *{action} Hit* — `{sym}` {d}\n"
                        f"Price: `{price}` | PnL: `{pnl_now:+.2f}%`\n"
                        f"_{extra}_ — still tracking TP3"
                    )
                    logger.info("%s hit %s @ %s", sym, action, price)

                else:  # SL or TP3
                    if action == "SL":
                        emoji = "❌"
                        reason_str = f"SL{_sl_context(t)}"
                    else:
                        emoji = "✅"
                        reason_str = "TP3"

                    tps = []
                    if t.get("tp1_hit"): tps.append("TP1")
                    if t.get("tp2_hit"): tps.append("TP2")
                    tps_str = (f" _{'+'.join(tps)} hit before_"
                               if tps else "")
                    pnl_pct = t.get("pnl_pct", 0.0)
                    await send_raw_message(
                        f"{emoji} *Trade Closed* — `{sym}` {d}\n"
                        f"Reason: *{reason_str}* at `{price}`{tps_str}\n"
                        f"PnL: `{pnl_pct:+.2f}%`"
                    )
                    logger.info("Trade closed: %s %s @ %s (%.2f%%)",
                                sym, reason_str, price, pnl_pct)

        except Exception as exc:
            logger.error("Monitor loop error: %s", exc, exc_info=True)

        await asyncio.sleep(MONITOR_INTERVAL_SEC)
