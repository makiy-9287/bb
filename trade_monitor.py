"""
Trade monitor — limit fills, TP1→TP2→TP3 tracking, no SL move after TP1.
Fixed: KeyError on 'tp_hits' for legacy trades.
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


def _load(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as exc:
        logger.warning("Read %s failed: %s", path, exc)
        return []


def _save(path: str, data: List[Dict[str, Any]]):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


# ── accessors ────────────────────────────────────────────────────
def get_pending_trades() -> List[Dict[str, Any]]:
    return [t for t in _load(ACTIVE_TRADES_FILE)
            if t.get("status") == "pending"]


def get_active_trades() -> List[Dict[str, Any]]:
    return [t for t in _load(ACTIVE_TRADES_FILE)
            if t.get("status") in ("open", "tp1_hit", "tp2_hit")]


def get_closed_trades() -> List[Dict[str, Any]]:
    return _load(CLOSED_TRADES_FILE)


def calc_pnl(trade: Dict[str, Any], price: float) -> Dict[str, float]:
    e = float(trade["entry"])
    if e == 0:
        return {"pnl_pct": 0.0}
    if trade["direction"].upper() == "LONG":
        pct = (price - e) / e * 100
    else:
        pct = (e - price) / e * 100
    return {"pnl_pct": pct}


async def get_prices(symbols: List[str]) -> Dict[str, float]:
    if not symbols:
        return {}
    url = f"{BINANCE_FUTURES_REST}/fapi/v1/ticker/price"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=10) as r:
            data = await r.json()
    pm = {i["symbol"]: float(i["price"]) for i in data}
    return {s: pm[s] for s in symbols if s in pm}


async def _get_1m_hl(symbols: List[str]) -> Dict[str, Dict[str, float]]:
    """Fetch last 2×1m candles — returns max high / min low / last close."""
    out: Dict[str, Dict[str, float]] = {}
    if not symbols:
        return out
    url = f"{BINANCE_FUTURES_REST}/fapi/v1/klines"
    async with aiohttp.ClientSession() as session:
        for sym in symbols:
            try:
                params = {"symbol": sym, "interval": "1m", "limit": 2}
                async with session.get(url, params=params, timeout=5) as r:
                    k = await r.json()
                if isinstance(k, list) and k:
                    out[sym] = {
                        "high": max(float(x[2]) for x in k),
                        "low":  min(float(x[3]) for x in k),
                        "close": float(k[-1][4]),
                    }
            except Exception:
                continue
    return out


def _key(t: Dict[str, Any]) -> str:
    return f"{t.get('symbol','?')}|{t.get('opened_at','')}"


def _append_closed(trade: Dict[str, Any]):
    closed = _load(CLOSED_TRADES_FILE)
    closed.append(trade)
    _save(CLOSED_TRADES_FILE, closed)


async def add_trade(signal: Dict[str, Any]):
    async with _get_lock():
        trades = _load(ACTIVE_TRADES_FILE)
        t = dict(signal)
        t["opened_at"] = datetime.utcnow().isoformat()
        t["status"] = "pending" if t.get("entry_type") == "LIMIT" else "open"
        t["tp_hits"] = []  # always initialize
        trades.append(t)
        _save(ACTIVE_TRADES_FILE, trades)
    logger.info("Trade added: %s (%s)", signal["symbol"], t["status"])


async def monitor_loop():
    logger.info("Trade monitor started (interval=%ds)", MONITOR_INTERVAL_SEC)

    while True:
        try:
            trades = _load(ACTIVE_TRADES_FILE)
            pending = [t for t in trades if t.get("status") == "pending"]
            active = [t for t in trades
                      if t.get("status") in ("open", "tp1_hit", "tp2_hit")]

            if not pending and not active:
                await asyncio.sleep(MONITOR_INTERVAL_SEC)
                continue

            symbols = list({t["symbol"] for t in pending + active})
            data = await _get_1m_hl(symbols)

            fills: List[tuple] = []
            closures: List[tuple] = []
            tps: List[tuple] = []

            async with _get_lock():
                current = _load(ACTIVE_TRADES_FILE)
                remaining: List[Dict[str, Any]] = []

                for t in current:
                    st = t.get("status")
                    sym = t["symbol"]
                    d = t["direction"].upper()
                    pd = data.get(sym)

                    # Auto-init tp_hits for legacy trades
                    if "tp_hits" not in t:
                        t["tp_hits"] = []

                    if st == "pending":
                        if pd is None:
                            remaining.append(t)
                            continue
                        filled = (
                            (d == "LONG" and pd["low"] <= float(t["entry"]))
                            or (d == "SHORT" and pd["high"] >= float(t["entry"]))
                        )
                        if filled:
                            t["status"] = "open"
                            t["filled_at"] = datetime.utcnow().isoformat()
                            fills.append((sym, d, float(t["entry"])))
                        remaining.append(t)
                        continue

                    if st in ("open", "tp1_hit", "tp2_hit"):
                        if pd is None:
                            remaining.append(t)
                            continue

                        entry = float(t["entry"])
                        sl = float(t["sl"])
                        tp1, tp2, tp3 = (float(t["tp1"]), float(t["tp2"]),
                                         float(t["tp3"]))

                        sl_hit = (
                            (d == "LONG" and pd["low"] <= sl)
                            or (d == "SHORT" and pd["high"] >= sl)
                        )
                        if sl_hit:
                            t["status"] = "closed"
                            t["closed_at"] = datetime.utcnow().isoformat()
                            t["exit_price"] = sl
                            t["exit_reason"] = "SL"
                            t["pnl_pct"] = round(
                                calc_pnl(t, sl)["pnl_pct"], 4)
                            _append_closed(t)
                            closures.append((sym, "SL", sl,
                                             t["pnl_pct"]))
                            continue

                        if d == "LONG":
                            if st == "open" and pd["high"] >= tp1:
                                t["status"] = "tp1_hit"
                                t["tp_hits"].append("TP1")
                                tps.append((sym, "TP1", tp1))
                            elif st == "tp1_hit" and pd["high"] >= tp2:
                                t["status"] = "tp2_hit"
                                t["tp_hits"].append("TP2")
                                tps.append((sym, "TP2", tp2))
                            elif st == "tp2_hit" and pd["high"] >= tp3:
                                t["status"] = "closed"
                                t["closed_at"] = datetime.utcnow().isoformat()
                                t["exit_price"] = tp3
                                t["exit_reason"] = "TP3"
                                t["pnl_pct"] = round(
                                    calc_pnl(t, tp3)["pnl_pct"], 4)
                                _append_closed(t)
                                closures.append((sym, "TP3", tp3,
                                                 t["pnl_pct"]))
                                continue
                        else:  # SHORT
                            if st == "open" and pd["low"] <= tp1:
                                t["status"] = "tp1_hit"
                                t["tp_hits"].append("TP1")
                                tps.append((sym, "TP1", tp1))
                            elif st == "tp1_hit" and pd["low"] <= tp2:
                                t["status"] = "tp2_hit"
                                t["tp_hits"].append("TP2")
                                tps.append((sym, "TP2", tp2))
                            elif st == "tp2_hit" and pd["low"] <= tp3:
                                t["status"] = "closed"
                                t["closed_at"] = datetime.utcnow().isoformat()
                                t["exit_price"] = tp3
                                t["exit_reason"] = "TP3"
                                t["pnl_pct"] = round(
                                    calc_pnl(t, tp3)["pnl_pct"], 4)
                                _append_closed(t)
                                closures.append((sym, "TP3", tp3,
                                                 t["pnl_pct"]))
                                continue

                        remaining.append(t)
                        continue

                    remaining.append(t)

                _save(ACTIVE_TRADES_FILE, remaining)

            for sym, d, px in fills:
                await send_raw_message(
                    f"✅ *Entry Filled* — `{sym}`\n"
                    f"Direction: {d}\n"
                    f"Fill: `{px}`"
                )
                logger.info("Limit filled: %s %s @ %s", sym, d, px)

            for sym, tp, px in tps:
                await send_raw_message(
                    f"🎯 *{tp} Hit* — `{sym}`\n"
                    f"Price: `{px}`"
                )
                logger.info("%s hit: %s @ %s", tp, sym, px)

            for sym, reason, px, pnl in closures:
                emoji = "✅" if reason.startswith("TP") else "❌"
                await send_raw_message(
                    f"{emoji} *Trade Closed* — `{sym}`\n"
                    f"Reason: *{reason}* at `{px}`\n"
                    f"PnL: `{pnl:+.2f}%`"
                )
                logger.info("Trade closed: %s %s @ %s (%.2f%%)",
                            sym, reason, px, pnl)

        except Exception as exc:
            logger.error("Monitor error: %s", exc, exc_info=True)

        await asyncio.sleep(MONITOR_INTERVAL_SEC)
