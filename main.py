"""
Scalp hybrid agent.

Every 5m candle close (Python only, no LLM):
  60 liquid USDT perps → 1h bias · 15m setup/liquidity · 5m entry confirmation → fully-confirmed setups.
If a scan has confirmed setups, they go (as ONE payload) to the NEXT api key in a 1→15→1 round-robin;
the model call runs in the background so scanning never blocks. The model picks the best (≤1).
Accepted signal → fresh-price check → Telegram → trade tracking. No time window: /on /off control it.
"""
import asyncio
import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config import (TOP_COINS_LIMIT, SCAN_INTERVAL_MIN, SCAN_OFFSET_SEC, USE_SESSION_WINDOW, START_ENABLED,
                    MAX_AGENT_BATCH, SETUP_COOLDOWN_MIN, SIGNAL_MAX_AGE_SEC, KEY_FAILOVER_TRIES, TIMEFRAMES,
                    OPENROUTER_API_KEYS, MIN_RANK_SCORE, COOLDOWN_MIN_AFTER_CLOSE, ENGINE_ONLY_MODE)
from ohlcv import TF_MS
from data_fetcher import get_universe, get_server_time_ms, get_last_prices, fetch_bundle
from ict_engine import analyze_symbol, market_bias
from payload_builder import build_candidate_payload
from agent import run_group
from risk import portfolio_allows
from telegram_sender import send_signal
from telegram_bot import telegram_bot_loop, update_state, AGENT_STATE
from trade_monitor import monitor_loop, add_trade, get_pending_trades, get_active_trades, recently_closed_symbols

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler("agent.log", encoding="utf-8")])
logger = logging.getLogger("main")

_SEEN: Dict[str, float] = {}            # setup fingerprint → time sent to a model
_INFLIGHT: set = set()                  # symbols currently under model review
_FRAME_CACHE: Dict[str, Dict[str, Any]] = {}   # sym → tf → frame (1h / 15m reused until a new candle closes)
_RR = {"next": 0}                        # round-robin pointer over API keys
_BG: set = set()                         # keep references to background tasks


def fingerprint(c: Dict[str, Any]) -> str:
    e = c["evidence"]
    raw = f"{c['symbol']}|{c['direction']}|{e['sweep']['sweep_ts']}|{e['mss']['break_ts']}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _purge_seen(now: float):
    for k in [k for k, t in _SEEN.items() if now - t > SETUP_COOLDOWN_MIN * 60]:
        _SEEN.pop(k, None)


def next_key_order(n_keys: int) -> List[int]:
    """Assigned key first, then the following keys (bounded failover). Pointer advances by one per setup-scan."""
    if n_keys <= 0:
        return []
    start = _RR["next"] % n_keys
    _RR["next"] = (start + 1) % n_keys
    return [(start + i) % n_keys for i in range(min(KEY_FAILOVER_TRIES, n_keys))]


async def _frames_for(symbols: List[str], now_ms: int) -> Dict[str, Dict[str, Any]]:
    """1h / 15m are re-fetched only when a newer candle has closed; 5m is fetched every scan."""
    need: Dict[str, List[str]] = {}
    for s in symbols:
        for tf in TIMEFRAMES:
            step = TF_MS[tf]
            expected_last_open = (now_ms // step) * step - step
            fr = _FRAME_CACHE.get(s, {}).get(tf)
            if fr is None or int(fr["t"].iloc[-1]) != expected_last_open:
                need.setdefault(tf, []).append(s)
    for tf, syms in need.items():
        frames, _ = await fetch_bundle(syms, [tf], now_ms)
        for s, d in frames.items():
            _FRAME_CACHE.setdefault(s, {})[tf] = d[tf]
    return {s: _FRAME_CACHE.get(s, {}) for s in symbols}


async def scan_once() -> Dict[str, Any]:
    now_ms = await get_server_time_ms()
    busy = {t.get("symbol") for t in get_pending_trades() + get_active_trades()} | recently_closed_symbols(COOLDOWN_MIN_AFTER_CLOSE)
    uni = await get_universe(TOP_COINS_LIMIT)
    symbols = [u["symbol"] for u in uni]
    if "BTCUSDT" not in symbols:
        symbols.append("BTCUSDT")
    frames = await _frames_for(symbols, now_ms)
    prices = await get_last_prices(symbols)

    btc = market_bias(frames["BTCUSDT"]) if len(frames.get("BTCUSDT", {})) == len(TIMEFRAMES) else None

    def _analyse():
        out, rej = [], {}
        for u in uni:
            s = u["symbol"]
            if s in busy or s in _INFLIGHT or len(frames.get(s, {})) < len(TIMEFRAMES) or s not in prices:
                continue
            r = analyze_symbol(s, frames[s], prices[s], btc)
            if r["candidate"]:
                out.append(r["candidate"])
            else:
                for k, v in r["rejects"].items():
                    rej[v.split(" (")[0][:60]] = rej.get(v.split(" (")[0][:60], 0) + 1
        return out, rej
    cands, rej = await asyncio.to_thread(_analyse)
    cands = [c for c in cands if c["score"] >= MIN_RANK_SCORE]
    now = time.time()
    _purge_seen(now)
    fresh = [c for c in cands if fingerprint(c) not in _SEEN]
    fresh.sort(key=lambda c: (-c["score"], -c["net_rr"], c["symbol"]))
    batch = fresh[:MAX_AGENT_BATCH]
    top_rej = sorted(rej.items(), key=lambda x: -x[1])[:4]
    logger.info("Scan: %d coins analysed · %d confirmed · %d new → batch %d | top rejects: %s", len(uni), len(cands),
                len(fresh), len(batch), "; ".join(f"{v}×{k}" for k, v in top_rej))
    return {"batch": batch, "frames": frames, "prices": prices, "n": len(uni), "confirmed": len(cands)}


async def review_and_publish(batch: List[Dict[str, Any]], frames: Dict[str, Dict[str, Any]], key_order: List[int],
                             scan_id: int):
    """Background: one payload → assigned key (failover to next keys) → validate → fresh-price check → Telegram."""
    syms = {c["symbol"] for c in batch}
    _INFLIGHT.update(syms)
    t0 = time.time()
    try:
        group = {"id": scan_id, "cands": {c["symbol"]: c for c in batch},
                 "payload": build_candidate_payload(batch, {c["symbol"]: frames[c["symbol"]] for c in batch})}
        result = None
        for ki in (key_order or [0]):
            result = await run_group(group, ki)
            if result["status"] not in ("failed", "invalid"):
                break
            logger.warning("Scan %d: key %d failed (%s) → failover", scan_id, ki + 1, result.get("error"))
        sig = result.get("signal") if result else None
        if not sig:
            return
        age = time.time() - t0
        px = (await get_last_prices([sig["symbol"]])).get(sig["symbol"])
        why = stale_reason(sig, px, age)
        if why:
            logger.warning("🚫 %s dropped before sending: %s", sig["symbol"], why)
            return
        ok, why = portfolio_allows(sig["direction"], sig["symbol"], get_pending_trades() + get_active_trades())
        if not ok:
            logger.info("🚫 %s dropped: %s", sig["symbol"], why)
            return
        if await send_signal(sig):
            await add_trade(sig)
            update_state(signal_count=AGENT_STATE["signal_count"] + 1, last_signal=sig)
    except Exception:
        logger.exception("review_and_publish crashed")
    finally:
        _INFLIGHT.difference_update(syms)


def stale_reason(sig: Dict[str, Any], px: Optional[float], age_sec: float) -> str:
    """Fresh-price gate: scalp setups die fast. Returns '' if still valid."""
    if age_sec > SIGNAL_MAX_AGE_SEC:
        return f"model answer is {age_sec:.0f}s old (> {SIGNAL_MAX_AGE_SEC}s)"
    if not px:
        return "no live price"
    long_ = sig["direction"] == "LONG"
    if (long_ and px <= sig["sl"]) or (not long_ and px >= sig["sl"]):
        return f"price {px} already beyond SL {sig['sl']}"
    if (long_ and px >= sig["tp1"]) or (not long_ and px <= sig["tp1"]):
        return f"price {px} already reached TP1 {sig['tp1']}"
    if sig["entry_type"] == "LIMIT":
        if (long_ and px < sig["entry"]) or (not long_ and px > sig["entry"]):
            return f"price {px} already through limit entry {sig['entry']}"
    elif abs(px - sig["entry"]) > 0.25 * sig["atr15"]:
        return f"CMP entry stale: price {px} vs entry {sig['entry']}"
    return ""


async def scan_loop():
    ev: asyncio.Event = AGENT_STATE["scan_trigger"]
    scan_id = 0
    while True:
        step = SCAN_INTERVAL_MIN * 60
        wait = step - (time.time() % step) + SCAN_OFFSET_SEC
        try:
            await asyncio.wait_for(ev.wait(), timeout=wait)
            logger.info("Manual /scan requested")
        except asyncio.TimeoutError:
            pass
        ev.clear()
        if not AGENT_STATE.get("enabled", True):
            continue
        if USE_SESSION_WINDOW:
            from schedule import in_window
            if not in_window():
                continue
        try:
            scan_id += 1
            update_state(last_scan_at=datetime.now(timezone.utc), scan_count=AGENT_STATE["scan_count"] + 1)
            res = await scan_once()
            if not res["batch"]:
                continue
            if not OPENROUTER_API_KEYS and not ENGINE_ONLY_MODE:
                logger.error("Confirmed setups found but no API keys configured")
                continue
            now = time.time()
            for c in res["batch"]:
                _SEEN[fingerprint(c)] = now
            order = next_key_order(len(OPENROUTER_API_KEYS))
            logger.info("→ Scan %d: %d setup(s) → API key %s (round-robin)", scan_id, len(res["batch"]),
                        order[0] + 1 if order else "-")
            task = asyncio.create_task(review_and_publish(res["batch"], res["frames"], order, scan_id))
            _BG.add(task)
            task.add_done_callback(_BG.discard)
        except Exception:
            logger.exception("scan failed")


async def main():
    AGENT_STATE["scan_trigger"] = asyncio.Event()
    update_state(started_at=datetime.now(timezone.utc), enabled=START_ENABLED, scan_count=0, signal_count=0)
    logger.info("Scalp hybrid agent: %d API key(s) · scan every %dm · enabled=%s · session window=%s",
                len(OPENROUTER_API_KEYS), SCAN_INTERVAL_MIN, START_ENABLED, USE_SESSION_WINDOW)
    await asyncio.gather(scan_loop(), monitor_loop(), telegram_bot_loop())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped")
