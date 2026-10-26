"""
V10 — Support/Resistance agent (4h · 1h · 15m).

Every 15 minutes (12:00–19:00 Asia/Colombo, /on /off): top-60 liquid perps → closed candles (4h×200, 1h×200, 15m×150)
→ groups of 4 coins → one API call per group (group i → key i) → the PROMPT decides; each call returns ≤1 best setup
→ a judge call picks the single best → Telegram (ADMIN ONLY) → simulated tracking.
The script never filters setups by strategy; it only skips coins that already have a live trade / were just signalled.
"""
import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from config import (TOP_COINS_LIMIT, SCAN_INTERVAL_MIN, SCAN_OFFSET_SEC, START_ENABLED, TIMEFRAMES, COINS_PER_CALL,
                    DEDUPE_HOURS, OPENROUTER_API_KEYS)
from ohlcv import TF_MS
from schedule import in_window, secs_to_next_scan, next_window_start
from data_fetcher import get_universe, get_server_time_ms, get_last_prices, fetch_bundle
from payload_builder import build_group_text
from agent import run_agents, judge_best
from telegram_sender import send_signal
from telegram_bot import telegram_bot_loop, update_state, AGENT_STATE
from trade_monitor import monitor_loop, add_trade, get_pending_trades, get_active_trades

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler("agent.log", encoding="utf-8")])
logger = logging.getLogger("main")

_SENT: Dict[str, float] = {}                    # symbol → time it was last signalled (repeat protection)
_FRAME_CACHE: Dict[str, Dict[str, Any]] = {}    # 4h / 1h reused until a newer candle closes
_SCAN_LOCK = asyncio.Lock()


def chunk(items: List[str], n: int) -> List[List[str]]:
    return [items[i:i + n] for i in range(0, len(items), n)]


def scan_allowed() -> bool:
    return AGENT_STATE.get("enabled", True) and in_window()


async def _frames_for(symbols: List[str], now_ms: int) -> Dict[str, Dict[str, Any]]:
    """Re-fetch a timeframe only when a newer closed candle exists; otherwise reuse the cache."""
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


async def build_groups(now_ms: int) -> List[Dict[str, Any]]:
    busy = {t.get("symbol") for t in get_pending_trades() + get_active_trades()}
    now = time.time()
    recent = {s for s, t in _SENT.items() if now - t < DEDUPE_HOURS * 3600}
    skip = busy | recent
    uni = await get_universe(TOP_COINS_LIMIT + len(skip))
    symbols = [u["symbol"] for u in uni if u["symbol"] not in skip][:TOP_COINS_LIMIT]
    if skip:
        logger.info("Skipping %d coin(s) with a live trade / recent signal", len(skip))
    frames = await _frames_for(symbols, now_ms)
    prices = await get_last_prices(symbols)
    ok = [s for s in symbols if s in prices and all(tf in frames.get(s, {}) for tf in TIMEFRAMES)]
    if len(ok) < len(symbols):
        logger.warning("%d coin(s) skipped: incomplete/invalid candle data", len(symbols) - len(ok))
    groups = []
    for i, syms in enumerate(chunk(ok, COINS_PER_CALL), start=1):
        groups.append({"id": i, "symbols": syms, "text": build_group_text(syms, {s: frames[s] for s in syms}, prices, now_ms)})
    return groups


async def scan_once(scan_id: int) -> Dict[str, Any]:
    now_ms = await get_server_time_ms()
    groups = await build_groups(now_ms)
    if not groups:
        logger.warning("Scan %d: nothing to analyse", scan_id)
        return {"best": None, "info": "no groups"}
    if not OPENROUTER_API_KEYS:
        logger.error("No OpenRouter API keys configured")
        return {"best": None, "info": "no keys"}
    logger.info("Scan %d: %d coins → %d group(s) → keys 1..%d", scan_id, sum(len(g["symbols"]) for g in groups),
                len(groups), min(len(groups), len(OPENROUTER_API_KEYS)))
    res = await run_agents(groups)
    best, how = await judge_best(res["candidates"], start_key=scan_id)
    if best is None:
        logger.info("Scan %d: no setup this time", scan_id)
        return {"best": None, "info": how}
    logger.info("Scan %d: best = %s %s (%s)", scan_id, best["direction"], best["symbol"], how)
    if await send_signal(best):
        _SENT[best["symbol"]] = time.time()
        await add_trade(best)
        update_state(signal_count=AGENT_STATE["signal_count"] + 1, last_signal=best)
    return {"best": best, "info": how}


async def _guarded_scan(scan_id: int):
    if _SCAN_LOCK.locked():
        logger.warning("Scan %d skipped: previous scan still running", scan_id)
        return
    async with _SCAN_LOCK:
        try:
            update_state(last_scan_at=datetime.now(timezone.utc), scan_count=AGENT_STATE["scan_count"] + 1)
            await scan_once(scan_id)
        except Exception:
            logger.exception("scan %d failed", scan_id)


async def scan_loop():
    ev: asyncio.Event = AGENT_STATE["scan_trigger"]
    scan_id, tasks = 0, set()
    while True:
        try:
            await asyncio.wait_for(ev.wait(), timeout=secs_to_next_scan(SCAN_INTERVAL_MIN, offset_sec=SCAN_OFFSET_SEC))
            logger.info("Manual /scan requested")
        except asyncio.TimeoutError:
            pass
        ev.clear()
        if not scan_allowed():
            continue
        scan_id += 1
        t = asyncio.create_task(_guarded_scan(scan_id))
        tasks.add(t)
        t.add_done_callback(tasks.discard)


async def main():
    AGENT_STATE["scan_trigger"] = asyncio.Event()
    update_state(started_at=datetime.now(timezone.utc), enabled=START_ENABLED, scan_count=0, signal_count=0)
    nxt = next_window_start()
    logger.info("V10 S/R agent · %d key(s) · every %dm · window 12:00–19:00 Asia/Colombo · enabled=%s · next window %s",
                len(OPENROUTER_API_KEYS), SCAN_INTERVAL_MIN, START_ENABLED, nxt.strftime("%a %H:%M"))
    await asyncio.gather(scan_loop(), monitor_loop(), telegram_bot_loop())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped")
