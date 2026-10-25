"""
Signal tracking (SIMULATED — not exchange fills).

States: pending → open → tp1_hit → tp2_hit → closed   |   pending → closed (EXPIRED / INVALIDATED)
Rules (all conservative, evaluated on CLOSED 1m candles strictly after signal creation):
  • candles that opened before the signal are never used (no pre-entry highs/lows);
  • LIMIT fills when price touches the entry; TP/SL are NOT evaluated on the fill candle except that an SL touch
    there is booked as an (ambiguous) stop-out;
  • TP and SL inside the same 1m candle → outcome is AMBIGUOUS and booked as the stop (never the favourable path);
  • partial exits follow TP_SPLIT; costs/funding/R are computed in performance.py.
"""
import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set

import aiohttp

from config import (BINANCE_FUTURES_REST, MONITOR_INTERVAL_SEC, ACTIVE_TRADES_FILE, CLOSED_TRADES_FILE,
                    TP_SPLIT, SIGNAL_EXPIRY_MIN, MAX_HOLD_HOURS, MOVE_SL_TO_BE_AFTER_TP1)
from performance import compute_result
from telegram_sender import send_raw_message

logger = logging.getLogger(__name__)
M1 = 60_000
_lock: Optional[asyncio.Lock] = None


def _get_lock() -> asyncio.Lock:
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    return _lock


def _load(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, list) else []
    except Exception as exc:
        logger.error("Could not read %s: %s", path, exc)
        return []


def _save(path: str, data: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def get_pending_trades() -> List[Dict[str, Any]]:
    return [t for t in _load(ACTIVE_TRADES_FILE) if t.get("status") == "pending"]


def get_active_trades() -> List[Dict[str, Any]]:
    return [t for t in _load(ACTIVE_TRADES_FILE) if t.get("status") in ("open", "tp1_hit", "tp2_hit")]


def get_closed_trades() -> List[Dict[str, Any]]:
    return _load(CLOSED_TRADES_FILE)


def recently_closed_symbols(minutes: float) -> Set[str]:
    cut = (time.time() * 1000) - minutes * 60_000
    return {t["symbol"] for t in _load(CLOSED_TRADES_FILE)
            if t.get("fill_price") and t.get("closed_ms", 0) >= cut}


def calc_pnl(trade: Dict[str, Any], price: float) -> Dict[str, float]:
    """Unrealised + realised gross PnL % (informational)."""
    fill = float(trade.get("fill_price") or trade["entry"])
    d = 1.0 if trade["direction"].upper() == "LONG" else -1.0
    realised = sum(l["frac"] * d * (l["price"] - fill) / fill for l in trade.get("legs", []))
    rem = float(trade.get("remaining", 1.0))
    return {"pnl_pct": (realised + rem * d * (price - fill) / fill) * 100}


# ───────────────────────────── pure state machine ─────────────────────────────
def _fav(d: float, c: List[float], x: float) -> bool:      # candle = [t,o,h,l,c]
    return c[2] >= x if d > 0 else c[3] <= x


def _adv(d: float, c: List[float], x: float) -> bool:
    return c[3] <= x if d > 0 else c[2] >= x


def _hist(t: Dict[str, Any], state: str, ts: int, price: float):
    t.setdefault("history", []).append({"state": state, "ts": ts, "price": price})


def _close(t: Dict[str, Any], outcome: str, ts: int, price: Optional[float], events: List[str], ambiguous=False):
    if price is not None and t.get("remaining", 0) > 1e-9:
        t["legs"].append({"price": price, "frac": t["remaining"], "reason": outcome, "ts": ts})
        t["remaining"] = 0.0
    t["status"], t["outcome"], t["closed_ms"] = "closed", outcome, ts
    if ambiguous:
        t["ambiguous"] = True
    _hist(t, "closed:" + outcome, ts, price or 0.0)
    events.append(outcome)


def advance_trade(t: Dict[str, Any], candles: List[List[float]], step_ms: int = M1) -> List[str]:
    """Process closed candles (chronological; 1m live, any step in backtests) for ONE trade. Mutates t, returns events."""
    events: List[str] = []
    d = 1.0 if str(t["direction"]).upper() == "LONG" else -1.0
    for c in candles:
        ts = int(c[0])
        if ts < t["next_ts"] or t["status"] == "closed":
            continue
        t["next_ts"] = ts + step_ms
        if t["status"] == "pending":
            if ts >= t["expires_ms"]:
                _close(t, "EXPIRED", ts, None, events)
                break
            touched = _adv(d, c, t["entry"])            # LONG: low<=entry ; SHORT: high>=entry
            if not touched:
                if _fav(d, c, t["tp1"]):                # target reached without us
                    _close(t, "INVALIDATED", ts, None, events)
                    break
                continue
            t.update(status="open", fill_price=t["entry"], filled_ms=ts, remaining=1.0)
            _hist(t, "filled", ts, t["entry"])
            events.append("FILLED")
            if _adv(d, c, t["sl"]):                     # same candle reaches the stop → ambiguous stop-out
                _close(t, "SL", ts, t["sl"], events, ambiguous=True)
                break
            continue                                    # TPs are not evaluated on the fill candle
        # open / partially exited
        if ts - t["filled_ms"] >= MAX_HOLD_HOURS * 3_600_000:
            _close(t, "TIME", ts, c[1], events)
            break
        while t["status"] != "closed":
            idx = {"open": 0, "tp1_hit": 1, "tp2_hit": 2}[t["status"]]
            tp = t[f"tp{idx + 1}"]
            hit_tp, hit_sl = _fav(d, c, tp), _adv(d, c, t["sl"])
            if hit_sl and hit_tp:
                _close(t, "SL", ts, t["sl"], events, ambiguous=True)      # never assume the favourable path
            elif hit_sl:
                _close(t, "SL" if abs(t["sl"] - t["fill_price"]) / t["fill_price"] > 1e-9 or not t.get("be") else "BE",
                       ts, t["sl"], events)
            elif hit_tp:
                frac = min(TP_SPLIT[idx], t["remaining"]) if idx < 2 else t["remaining"]
                t["legs"].append({"price": tp, "frac": frac, "reason": f"TP{idx + 1}", "ts": ts})
                t["remaining"] = max(0.0, t["remaining"] - frac)
                events.append(f"TP{idx + 1}")
                if idx == 2:
                    _close(t, "TP3", ts, None, events)
                else:
                    t["status"] = f"tp{idx + 1}_hit"
                    _hist(t, t["status"], ts, tp)
                    if idx == 0 and MOVE_SL_TO_BE_AFTER_TP1:
                        t["sl"], t["be"] = t["fill_price"], True
                        if _adv(d, c, t["sl"]):         # BE touched in the same candle → conservative
                            _close(t, "BE", ts, t["sl"], events, ambiguous=True)
                    continue
            break
        if t["status"] == "closed":
            break
    if t["status"] == "closed":
        t["result"] = compute_result(t)
    return events


# ───────────────────────────── IO ─────────────────────────────
async def get_prices(symbols: List[str]) -> Dict[str, float]:
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{BINANCE_FUTURES_REST}/fapi/v1/ticker/price", timeout=aiohttp.ClientTimeout(total=10)) as r:
            data = await r.json(content_type=None)
    return {i["symbol"]: float(i["price"]) for i in data if i.get("symbol") in set(symbols)}


async def _closed_1m(session: aiohttp.ClientSession, symbol: str, start_ms: int) -> List[List[float]]:
    async with session.get(f"{BINANCE_FUTURES_REST}/fapi/v1/klines",
                           params={"symbol": symbol, "interval": "1m", "startTime": start_ms, "limit": 1000},
                           timeout=aiohttp.ClientTimeout(total=10)) as r:
        data = await r.json(content_type=None)
    if not isinstance(data, list):
        raise RuntimeError(f"klines error: {str(data)[:120]}")
    now = int(time.time() * 1000)
    return [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4])] for k in data if int(k[0]) + M1 <= now]


def build_trade_record(signal: Dict[str, Any], now_ms: int, step_ms: int = M1) -> Dict[str, Any]:
    """Initial tracked-trade record (shared by the live monitor and the backtest)."""
    cmp_ = signal["entry_type"] == "CMP"
    t = {
        "id": f"{signal['symbol']}-{now_ms}", "symbol": signal["symbol"], "direction": signal["direction"],
        "entry_type": signal["entry_type"], "entry": float(signal["entry"]), "sl": float(signal["sl"]),
        "sl_initial": float(signal["sl"]), "tp1": float(signal["tp1"]), "tp2": float(signal["tp2"]),
        "tp3": float(signal["tp3"]), "zone_low": signal.get("zone_low"), "zone_high": signal.get("zone_high"),
        "score": signal.get("score"), "net_rr": signal.get("net_rr"), "setup_type": signal.get("setup_type"),
        "ambiguous": False, "created_ms": now_ms, "next_ts": -(-now_ms // step_ms) * step_ms,
        "expires_ms": now_ms + SIGNAL_EXPIRY_MIN * 60_000, "legs": [], "history": [],
        "status": "open" if cmp_ else "pending", "remaining": 1.0 if cmp_ else 0.0,
        "fill_price": float(signal["entry"]) if cmp_ else None, "filled_ms": now_ms if cmp_ else None,
    }
    _hist(t, "proposed", now_ms, t["entry"])
    _hist(t, t["status"], now_ms, t["entry"])
    return t


async def add_trade(signal: Dict[str, Any]):
    now = int(time.time() * 1000)
    async with _get_lock():
        data = _load(ACTIVE_TRADES_FILE)
        if any(x["symbol"] == signal["symbol"] and x.get("status") != "closed" for x in data):
            logger.info("Trade for %s already tracked — duplicate ignored", signal["symbol"])
            return
        t = build_trade_record(signal, now)
        data.append(t)
        _save(ACTIVE_TRADES_FILE, data)
    logger.info("Tracking %s %s (%s)", t["symbol"], t["direction"], t["status"])


def _msg(t: Dict[str, Any], ev: str) -> str:
    s, d = t["symbol"], t["direction"]
    if ev == "FILLED":
        return f"🟡➡️🟢 *{s}* {d} LIMIT filled @ `{t['fill_price']:.6g}` _(simulated)_"
    if ev in ("TP1", "TP2", "TP3"):
        return f"🎯 *{s}* {d} {ev} hit @ `{t[ev.lower()]:.6g}`"
    r = t.get("result", {})
    tail = (f"\nNet: `{r['net_pct']:+.2f}%` · `{r['r_multiple']:+.2f}R` _(after fees/slippage/funding, simulated)_"
            if r.get("filled") else "")
    amb = "\n⚠️ TP and SL inside one 1m candle — booked conservatively as stop" if t.get("ambiguous") else ""
    names = {"SL": "🛑 stop hit", "BE": "⚪ stopped at break-even", "TP3": "🏁 all targets hit", "EXPIRED": "⌛ limit expired unfilled",
             "INVALIDATED": "🚫 invalidated (target reached before fill)", "TIME": "⏱ closed on max hold time"}
    return f"{names.get(ev, ev)} — *{s}* {d}{tail}{amb}"


async def monitor_loop():
    logger.info("Trade monitor started (every %ss)", MONITOR_INTERVAL_SEC)
    while True:
        try:
            async with _get_lock():
                data = _load(ACTIVE_TRADES_FILE)
            if data:
                out_msgs: List[str] = []
                async with aiohttp.ClientSession() as sess:
                    for t in data:
                        if t.get("status") == "closed":
                            continue
                        try:
                            cs = await _closed_1m(sess, t["symbol"], t["next_ts"])
                        except Exception as exc:
                            logger.warning("1m fetch %s failed: %s", t["symbol"], exc)
                            continue
                        for ev in advance_trade(t, cs):
                            out_msgs.append(_msg(t, ev))
                async with _get_lock():
                    cur = {x["id"]: x for x in _load(ACTIVE_TRADES_FILE)}
                    for t in data:                         # merge: keep trades added while we were fetching
                        cur[t["id"]] = t
                    keep, closed = [], _load(CLOSED_TRADES_FILE)
                    for t in cur.values():
                        (closed if t.get("status") == "closed" else keep).append(t)
                    _save(ACTIVE_TRADES_FILE, keep)
                    _save(CLOSED_TRADES_FILE, closed)
                for m in out_msgs:
                    await send_raw_message(m)
        except Exception:
            logger.exception("monitor error")
        await asyncio.sleep(MONITOR_INTERVAL_SEC)
