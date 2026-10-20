"""
v4.2 Main orchestrator — Batched single-request mode.
Fetch 40 coins, process in batches of 10.
"""
import asyncio
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from config import (
    SLST_START, SLST_END, SLST_UTC_OFFSET,
    TOP_COINS_LIMIT, REASONING_EFFORT, DEEPSEEK_MODEL,
    SCAN_INTERVAL_MIN, OFF_WEEKDAYS, DAY_NAMES,
    CANDLE_COUNTS, MONITOR_INTERVAL_SEC, BATCH_SIZE
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
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("agent.log")],
)
logging.getLogger("pandas_ta_classic").setLevel(logging.ERROR)
logging.getLogger("pandas_ta_classic.utils.core").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger("main")
SLST_TZ = timezone(timedelta(hours=SLST_UTC_OFFSET))

_CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}

def in_execution_window() -> bool:
    now = datetime.now(SLST_TZ)
    if now.weekday() in OFF_WEEKDAYS:
        return False
    return SLST_START <= now.time() < SLST_END

def _next_window_start(now: datetime) -> datetime:
    c = now.replace(hour=SLST_START.hour, minute=SLST_START.minute,
                    second=0, microsecond=0)
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

def _signal_score(sig: Dict[str, Any]) -> tuple:
    conf = _CONFIDENCE_RANK.get(sig.get("confidence", "low"), 1)
    try:
        entry = float(sig["entry"]); sl = float(sig["sl"]); tp3 = float(sig["tp3"])
        risk = abs(entry - sl); reward = abs(tp3 - entry)
        rr = reward / risk if risk > 0 else 0.0
    except Exception:
        rr = 0.0
    return (conf, rr)

def _pick_best_signal(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
    return max(signals, key=_signal_score)


async def signal_cycle():
    cycle_start = time.time()
    logger.info("═══ Signal cycle starting ═══")

    try:
        # 1. Fetch Top 40 Symbols
        logger.info("Step 1/5 — Fetching top %d symbols …", TOP_COINS_LIMIT)
        t = time.time()
        symbols = await get_top_volume_symbols(TOP_COINS_LIMIT)
        if not symbols:
            logger.warning("No symbols — cycle aborted")
            return
        logger.info("Step 1/5 — Done: %d symbols in %.1fs", len(symbols), time.time() - t)

        # 2. Fetch OHLCV for all 40
        logger.info("Step 2/5 — Fetching OHLCV …")
        t = time.time()
        bundle = await fetch_all_data(symbols)
        logger.info("Step 2/5 — Done: %d symbols in %.1fs", len(bundle), time.time() - t)

        # 3. Compute Indicators for all 40
        logger.info("Step 3/5 — Computing indicators …")
        t = time.time()
        all_data = []
        for sym, tf_data in bundle.items():
            try:
                all_data.append(compute_multi_timeframe(sym, tf_data))
            except Exception as exc:
                logger.warning("Indicator error %s: %s", sym, exc)
        logger.info("Step 3/5 — Done: %d symbols in %.1fs", len(all_data), time.time() - t)

        if not all_data:
            logger.warning("No data — cycle aborted")
            return

        # 4. Split into batches of 10
        batches = [all_data[i:i + BATCH_SIZE] for i in range(0, len(all_data), BATCH_SIZE)]
        logger.info("Step 4/5 — Split into %d batches of %d coins", len(batches), BATCH_SIZE)

        signals: List[Dict[str, Any]] = []

        # 5. Process each batch
        for i, batch_data in enumerate(batches, 1):
            logger.info("▶️ Processing Batch %d/%d (%d coins) …", i, len(batches), len(batch_data))
            payload = build_payload(batch_data)
            
            t_api = time.time()
            signal = await run_agent(payload)
            logger.info("Batch %d response in %.1fs", i, time.time() - t_api)

            if signal:
                logger.info("✅ Batch %d Signal: %s %s (%s)", i, signal["direction"], signal["symbol"], signal.get("confidence"))
                signals.append(signal)
            else:
                logger.info("✗ Batch %d returned NO_SETUP", i)

        # 6. Pick the Best Signal
        update_state(
            last_scan_at=datetime.now(SLST_TZ),
            scan_count=AGENT_STATE["scan_count"] + 1,
        )

        logger.info("━━━ Cycle summary ━━━")
        logger.info("Batches: %d | Signals: %d", len(batches), len(signals))

        if signals:
            best = _pick_best_signal(signals)
            logger.info("🏆 Best: %s %s (%s)", best["direction"], best["symbol"], best.get("confidence"))
            added = await add_trade(best)
            if added:
                update_state(
                    last_signal=best,
                    signal_count=AGENT_STATE["signal_count"] + 1,
                )
                logger.info("✓ Trade registered: %s", best["symbol"])
            else:
                logger.info("✗ Skipped — max positions reached")
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
            try:
                await signal_cycle()
            except Exception as exc:
                logger.error("Cycle failed: %s", exc, exc_info=True)
            wait_sec = _seconds_to_next_scan_boundary()
        else:
            wait_sec = _seconds_until_next_window()
            if now.weekday() in OFF_WEEKDAYS:
                logger.info("[%s] OFF-DAY — sleeping %s", day, _fmt_sleep(wait_sec))
            else:
                logger.info("[%s] Outside hours — sleeping %s", day, _fmt_sleep(wait_sec))

        try:
            await asyncio.wait_for(scan_event.wait(), timeout=wait_sec)
            scan_event.clear()
            logger.info("🚀 Manual /scan — running now")
        except asyncio.TimeoutError:
            pass


async def signal_loop_supervisor():
    while True:
        try:
            await signal_loop()
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Loop crashed: %s — restart in 30s", exc)
            await asyncio.sleep(30)


async def main():
    update_state(started_at=datetime.now(SLST_TZ))
    AGENT_STATE["scan_trigger"] = asyncio.Event()

    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║  Binance Futures AI Agent — v4.2 BATCHED    ║")
    logger.info("╚══════════════════════════════════════════════╝")
    logger.info("Model: %s | Reasoning: %s", DEEPSEEK_MODEL, REASONING_EFFORT)
    logger.info("Timeframes: 4H(%d) + 1H(%d) + 15m(%d) candles",
                CANDLE_COUNTS["4h"], CANDLE_COUNTS["1h"], CANDLE_COUNTS["15m"])
    logger.info("Coins: %d | Batch size: %d", TOP_COINS_LIMIT, BATCH_SIZE)
    logger.info("Trading hours: %s – %s SLST",
                SLST_START.strftime("%H:%M"), SLST_END.strftime("%H:%M"))
    logger.info("Trade monitor: 24/7 (%ds interval)", MONITOR_INTERVAL_SEC)

    results = await asyncio.gather(
        signal_loop_supervisor(),
        monitor_loop(),
        telegram_bot_supervisor(),
        return_exceptions=True,
    )
    for r in results:
        if isinstance(r, Exception):
            logger.error("Task crashed: %s", r, exc_info=r)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully …")
