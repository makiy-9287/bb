"""
v4.1 Main orchestrator — Single-request mode.

All 40 coins + raw candles sent in ONE DeepSeek call.
No batching, no charts.

Scan window: 12:00–20:00 SLST (Mon–Thu), every 15 min.
Trade monitor: 24/7, every 5s.
"""
import asyncio
import logging
import sys
import time
from datetime import datetime, timedelta, timezone

from config import (
    SLST_START, SLST_END, SLST_UTC_OFFSET,
    TOP_COINS_LIMIT, REASONING_EFFORT, DEEPSEEK_MODEL,
    SCAN_INTERVAL_MIN, OFF_WEEKDAYS, DAY_NAMES,
    CANDLE_COUNTS, MONITOR_INTERVAL_SEC,
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


# ── Time helpers ─────────────────────────────────────────────────
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


# ── Signal cycle ─────────────────────────────────────────────────
async def signal_cycle():
    cycle_start = time.time()
    logger.info("═══ Signal cycle starting ═══")

    try:
        # Step 1 — Fetch symbols
        logger.info("Step 1/4 — Fetching top symbols …")
        t = time.time()
        symbols = await get_top_volume_symbols(TOP_COINS_LIMIT)
        if not symbols:
            logger.warning("No symbols — cycle aborted")
            return
        logger.info("Step 1/4 — Done: %d symbols in %.1fs",
                    len(symbols), time.time() - t)

        # Step 2 — OHLCV
        logger.info("Step 2/4 — Fetching OHLCV (4H/1H/15m) …")
        t = time.time()
        bundle = await fetch_all_data(symbols)
        logger.info("Step 2/4 — Done: %d symbols in %.1fs",
                    len(bundle), time.time() - t)

        # Step 3 — Indicators
        logger.info("Step 3/4 — Computing indicators …")
        t = time.time()
        all_data = []
        for sym, tf_data in bundle.items():
            try:
                all_data.append(compute_multi_timeframe(sym, tf_data))
            except Exception as exc:
                logger.warning("Indicator error %s: %s", sym, exc)
        logger.info("Step 3/4 — Done: %d symbols in %.1fs",
                    len(all_data), time.time() - t)

        if not all_data:
            logger.warning("No data — cycle aborted")
            return

        # Step 4 — Single payload + single request
        logger.info("Step 4/4 — Building single payload for %d coins …",
                    len(all_data))
        payload = build_payload(all_data)
        logger.info("Payload: %.1f KB", len(payload) / 1024)

        logger.info("Sending to DeepSeek (model=%s, effort=%s) …",
                    DEEPSEEK_MODEL, REASONING_EFFORT)
        t = time.time()
        signal = await run_agent(payload)
        logger.info("Agent response in %.1fs", time.time() - t)

        # Update state
        update_state(
            last_scan_at=datetime.now(SLST_TZ),
            scan_count=AGENT_STATE["scan_count"] + 1,
        )

        # Handle signal
        if signal:
            logger.info("✅ Signal: %s %s (%s)",
                        signal["direction"], signal["symbol"],
                        signal.get("confidence"))
            added = await add_trade(signal)
            if added:
                update_state(
                    last_signal=signal,
                    signal_count=AGENT_STATE["signal_count"] + 1,
                )
                logger.info("✓ Trade registered: %s", signal["symbol"])
            else:
                logger.info("✗ Skipped — max positions reached")
        else:
            logger.info("✗ No valid setup this cycle")

    except Exception as exc:
        logger.error("Cycle crashed: %s", exc, exc_info=True)

    finally:
        logger.info("═══ Cycle complete in %.1fs ═══\n",
                    time.time() - cycle_start)


# ── Signal loop ──────────────────────────────────────────────────
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
                logger.info("[%s] OFF-DAY — sleeping %s",
                            day, _fmt_sleep(wait_sec))
            else:
                logger.info("[%s] Outside hours (%s SLST) — sleeping %s",
                            day, now.strftime("%H:%M"), _fmt_sleep(wait_sec))

        try:
            await asyncio.wait_for(scan_event.wait(), timeout=wait_sec)
            scan_event.clear()
            logger.info("🚀 Manual /scan — running now")
        except asyncio.TimeoutError:
            pass


async def signal_loop_supervisor():
    """Restart signal_loop if it crashes unexpectedly."""
    while True:
        try:
            await signal_loop()
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Signal loop crashed: %s — restart in 30s", exc)
            await asyncio.sleep(30)


# ── Main ─────────────────────────────────────────────────────────
async def main():
    update_state(started_at=datetime.now(SLST_TZ))
    AGENT_STATE["scan_trigger"] = asyncio.Event()

    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║  Binance Futures AI Agent — v4.1 SINGLE     ║")
    logger.info("╚══════════════════════════════════════════════╝")
    logger.info("Model: %s | Reasoning: %s", DEEPSEEK_MODEL, REASONING_EFFORT)
    logger.info("Timeframes: 4H(%d) + 1H(%d) + 15m(%d) candles",
                CANDLE_COUNTS["4h"], CANDLE_COUNTS["1h"], CANDLE_COUNTS["15m"])
    logger.info("Coins: %d | Mode: SINGLE REQUEST (no batches)",
                TOP_COINS_LIMIT)
    logger.info("Data: Raw candles + RSI/MA/ATR — no charts")
    logger.info("Trading days: Monday – Thursday")
    logger.info("Trading hours: %s – %s SLST",
                SLST_START.strftime("%H:%M"), SLST_END.strftime("%H:%M"))
    logger.info("Off days: Friday, Saturday, Sunday")
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
