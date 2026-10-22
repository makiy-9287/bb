"""
v6 Main Orchestrator — Parallel Multi-Key Execution.
"""
import asyncio
import logging
import sys
import time
from datetime import datetime, timedelta, timezone

from config import (
    SLST_START, SLST_END, SLST_UTC_OFFSET,
    TOP_COINS_LIMIT, BATCH_SIZE, OPENROUTER_API_KEYS, OPENROUTER_MODEL,
    SCAN_INTERVAL_MIN, OFF_WEEKDAYS, DAY_NAMES, MONITOR_INTERVAL_SEC
)
from data_fetcher import get_top_volume_symbols, fetch_all_data
from indicators import compute_multi_timeframe
from payload_builder import build_payload
from agent import run_agent
from trade_monitor import monitor_loop, add_trade
from telegram_bot import telegram_bot_supervisor, update_state, AGENT_STATE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("agent.log")],
)
logging.getLogger("pandas_ta_classic").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger("main")
SLST_TZ = timezone(timedelta(hours=SLST_UTC_OFFSET))

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

async def process_single_batch(batch_data, api_key, batch_idx):
    """Runs a single batch asynchronously."""
    payload = build_payload(batch_data)
    logger.info("▶️ Batch %d started (Key: ...%s)", batch_idx, api_key[-4:])
    
    t0 = time.time()
    signal = await run_agent(payload, api_key)
    elapsed = time.time() - t0
    
    if signal:
        logger.info("✅ Batch %d found: %s %s in %.1fs", batch_idx, signal["direction"], signal["symbol"], elapsed)
        return signal
    else:
        logger.info("✗ Batch %d returned NO_SETUP in %.1fs", batch_idx, elapsed)
        return None


async def signal_cycle():
    cycle_start = time.time()
    logger.info("═══ Signal cycle starting ═══")

    try:
        logger.info("Step 1/4 — Fetching top %d symbols …", TOP_COINS_LIMIT)
        symbols = await get_top_volume_symbols(TOP_COINS_LIMIT)
        if not symbols:
            return

        logger.info("Step 2/4 — Fetching OHLCV (1D/4H/1H/15M) …")
        bundle = await fetch_all_data(symbols)

        logger.info("Step 3/4 — Computing indicators …")
        all_data = []
        for sym, tf_data in bundle.items():
            try:
                all_data.append(compute_multi_timeframe(sym, tf_data))
            except Exception as exc:
                logger.warning("Indicator error %s: %s", sym, exc)

        if not all_data:
            return

        # ── Step 4: Batch & Parallel API Calls ──
        batches = [all_data[i:i + BATCH_SIZE] for i in range(0, len(all_data), BATCH_SIZE)]
        logger.info("Step 4/4 — Launching %d batches IN PARALLEL", len(batches))

        # Map each batch to a specific API Key
        tasks = []
        for i, batch in enumerate(batches):
            # Use modulo in case batches > keys
            key = OPENROUTER_API_KEYS[i % len(OPENROUTER_API_KEYS)]
            tasks.append(process_single_batch(batch, key, i+1))

        # 🚀 RUN ALL AT THE SAME TIME 🚀
        results = await asyncio.gather(*tasks)

        # Filter out None values
        signals = [s for s in results if s]

        update_state(last_scan_at=datetime.now(SLST_TZ), scan_count=AGENT_STATE["scan_count"] + 1)

        if signals:
            # Pick best signal based on score (13-15)
            best = max(signals, key=lambda x: x.get("", 0))
            logger.info("🏆 Best Signal: %s %s (Score: %d)", best["direction"], best["symbol"], best["score"])
            
            added = await add_trade(best)
            if added:
                update_state(last_signal=best, signal_count=AGENT_STATE["signal_count"] + 1)
        else:
            logger.info("✗ No valid setup found across all 15 batches")

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
            logger.info("[%s] Outside hours — sleeping %s", day, f"{int(wait_sec//3600)}h {int((wait_sec%3600)//60)}m")

        try:
            await asyncio.wait_for(scan_event.wait(), timeout=wait_sec)
            scan_event.clear()
        except asyncio.TimeoutError:
            pass

async def main():
    update_state(started_at=datetime.now(SLST_TZ))
    AGENT_STATE["scan_trigger"] = asyncio.Event()

    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║  Binance Agent — v6 PARALLEL ENGINE         ║")
    logger.info("╚══════════════════════════════════════════════╝")
    logger.info("Model: %s", OPENROUTER_MODEL)
    logger.info("Coins: %d | Batch size: %d (Total batches: %d)", TOP_COINS_LIMIT, BATCH_SIZE, TOP_COINS_LIMIT//BATCH_SIZE)
    logger.info("API Keys: %d", len(OPENROUTER_API_KEYS))
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
