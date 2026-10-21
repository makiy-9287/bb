"""
v5 Main orchestrator — Multi-Key Rotation.
"""
import asyncio
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from config import (
    SLST_START, SLST_END, SLST_UTC_OFFSET,
    TOP_COINS_LIMIT, BATCH_SIZE, OPENROUTER_API_KEYS, OPENROUTER_MODEL, # 👈 OPENROUTER_MODEL මෙතනට add කළා
    SCAN_INTERVAL_MIN, OFF_WEEKDAYS, DAY_NAMES,
    CANDLE_COUNTS, MONITOR_INTERVAL_SEC

)
from data_fetcher import get_top_volume_symbols, fetch_all_data
from indicators import compute_multi_timeframe
from payload_builder import build_payload
from agent import run_agent
from trade_monitor import monitor_loop, add_trade
from telegram_bot import telegram_bot_supervisor, update_state, AGENT_STATE
from key_manager import KeyManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("agent.log")],
)
logging.getLogger("pandas_ta_classic").setLevel(logging.ERROR)
logging.getLogger("pandas_ta_classic.utils.core").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger("main")
SLST_TZ = timezone(timedelta(hours=SLST_UTC_OFFSET))

# Initialize Key Manager
key_manager = KeyManager(OPENROUTER_API_KEYS)

def in_execution_window() -> bool:
    now = datetime.now(SLST_TZ)
    if now.weekday() in OFF_WEEKDAYS:
        return False
    return SLST_START <= now.time() < SLST_END

def _next_window_start(now: datetime) -> datetime:
    c = now.replace(hour=SLST_START.hour, minute=SLST_START.minute, second=0, microsecond=0)
    if c <= now or c.weekday() in OFF_WEEKDAYS:
        c += timedelta(days=1)
        while c.weekday() in OFF_WEEKDAYS:
            c += timedelta(days=1)
    return c

def _seconds_until_next_window() -> float:
    now = datetime.now(SLST_TZ)
    return max((_next_window_start(now) - now).total_seconds(), 60)

def _seconds_to_next_scan_boundary() -> float:
    now = datetime.now(SLST_TZ)
    m = SCAN_INTERVAL_MIN - (now.minute % SCAN_INTERVAL_MIN)
    nxt = (now + timedelta(minutes=m)).replace(second=0, microsecond=0)
    wait = (nxt - now).total_seconds()
    if wait < 30:
        wait += SCAN_INTERVAL_MIN * 60
    return wait

def _fmt_sleep(s: float) -> str:
    if s >= 3600:
        return f"{int(s // 3600)}h {int((s % 3600) // 60)}m"
    return f"{int(s // 60)}m"


async def signal_cycle():
    cycle_start = time.time()
    
    # ── ROTATE API KEY AT THE START OF EVERY SCAN ──
    current_api_key = key_manager.rotate()
    
    logger.info("═══ Signal cycle starting (Key: ...%s) ═══", current_api_key[-4:])

    try:
        logger.info("Step 1/4 — Fetching top %d symbols …", TOP_COINS_LIMIT)
        symbols = await get_top_volume_symbols(TOP_COINS_LIMIT)
        if not symbols:
            return

        logger.info("Step 2/4 — Fetching OHLCV (1D/4H/1H/15M) …")
        t = time.time()
        bundle = await fetch_all_data(symbols)
        logger.info("Step 2/4 — Done: %d symbols in %.1fs", len(bundle), time.time() - t)

        logger.info("Step 3/4 — Computing indicators …")
        t = time.time()
        all_data = []
        for sym, tf_data in bundle.items():
            try:
                all_data.append(compute_multi_timeframe(sym, tf_data))
            except Exception as exc:
                logger.warning("Indicator error %s: %s", sym, exc)
        logger.info("Step 3/4 — Done: %d symbols in %.1fs", len(all_data), time.time() - t)

        if not all_data:
            return

        # Split into batches (70 / 10 = 7 batches)
        batches = [all_data[i:i + BATCH_SIZE] for i in range(0, len(all_data), BATCH_SIZE)]
        logger.info("Step 4/4 — Split into %d batches of %d coins", len(batches), BATCH_SIZE)

        signals = []
        for i, batch_data in enumerate(batches, 1):
            logger.info("▶️ Processing Batch %d/%d (Key: ...%s) …", i, len(batches), current_api_key[-4:])
            payload = build_payload(batch_data)
            signal = await run_agent(payload, current_api_key)

            if signal:
                logger.info("✅ Batch %d found: %s %s", i, signal["direction"], signal["symbol"])
                signals.append(signal)

        update_state(last_scan_at=datetime.now(SLST_TZ), scan_count=AGENT_STATE["scan_count"] + 1)

        if signals:
            # Pick best signal
            def score(sig): return sig.get("score", 0)
            best = max(signals, key=score)
            logger.info("🏆 Best Signal: %s %s (Score: %d)", best["direction"], best["symbol"], best["score"])
            
            added = await add_trade(best)
            if added:
                update_state(last_signal=best, signal_count=AGENT_STATE["signal_count"] + 1)
        else:
            logger.info("✗ No valid setup this cycle")

    except Exception as exc:
        logger.error("Cycle crashed: %s", exc, exc_info=True)
    finally:
        logger.info("═══ Cycle complete in %.1fs ═══\n", time.time() - cycle_start)


async def signal_loop():
    scan_event: asyncio.Event = AGENT_STATE["scan_trigger"]

    while True:
        now = datetime.now(SLST_TZ)
        day = DAY_NAMES[now.weekday()]

        if in_execution_window():
            await signal_cycle()
            wait_sec = _seconds_to_next_scan_boundary()
        else:
            wait_sec = _seconds_until_next_window()
            logger.info("[%s] Outside hours — sleeping %s", day, _fmt_sleep(wait_sec))

        try:
            await asyncio.wait_for(scan_event.wait(), timeout=wait_sec)
            scan_event.clear()
        except asyncio.TimeoutError:
            pass

async def main():
    update_state(started_at=datetime.now(SLST_TZ))
    AGENT_STATE["scan_trigger"] = asyncio.Event()

    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║  Binance Agent — v5 OPENROUTER MULTI-KEY    ║")
    logger.info("╚══════════════════════════════════════════════╝")
    logger.info("Model: %s", OPENROUTER_MODEL)
    logger.info("Timeframes: 1D(50) + 4H(150) + 1H(100) + 15m(100)")
    logger.info("Coins: %d | Batch size: %d", TOP_COINS_LIMIT, BATCH_SIZE)
    logger.info("Total API Keys: %d", len(OPENROUTER_API_KEYS))
    logger.info("Trading hours: %s – %s SLST", SLST_START.strftime("%H:%M"), SLST_END.strftime("%H:%M"))

    await asyncio.gather(
        signal_loop(),
        monitor_loop(),
        telegram_bot_supervisor(),
        return_exceptions=True,
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully …")
