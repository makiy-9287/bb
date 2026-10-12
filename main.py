"""
Main orchestrator — DAY MODE (Mon–Thu, 13:00–20:00 SLST).
Trade monitor runs 24/7 regardless of window.
"""
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone

from config import (
    SLST_START, SLST_END, SLST_UTC_OFFSET,
    TOP_COINS_LIMIT, REASONING_EFFORT, DEEPSEEK_MODEL,
    SCAN_INTERVAL_MIN, OFF_WEEKDAYS, DAY_NAMES,
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
    """
    Seconds until the next 15-min boundary (e.g. :00, :15, :30, :45).
    If we're within 30s of the next boundary (implying we might have just
    finished a scan), skip to the following boundary to avoid double-firing.
    """
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
    logger.info("═══ Signal cycle starting ═══")
    symbols = await get_top_volume_symbols(TOP_COINS_LIMIT)
    logger.info("Fetching OHLCV for %d symbols …", len(symbols))
    bundle = await fetch_all_data(symbols)

    logger.info("Computing indicators …")
    all_data = []
    for sym, tf_data in bundle.items():
        try:
            all_data.append(compute_multi_timeframe(sym, tf_data))
        except Exception as exc:
            logger.warning("Indicator error %s: %s", sym, exc)

    if not all_data:
        logger.warning("No data — skipping cycle")
        return

    payload = build_payload(all_data)
    logger.info("Payload size: %.1f KB", len(payload) / 1024)

    logger.info("Sending to DeepSeek (model=%s, effort=%s) …",
                DEEPSEEK_MODEL, REASONING_EFFORT)
    signal = await run_agent(payload)

    update_state(
        last_scan_at=datetime.now(SLST_TZ),
        scan_count=AGENT_STATE["scan_count"] + 1,
    )

    if signal:
        await add_trade(signal)
        update_state(
            last_signal=signal,
            signal_count=AGENT_STATE["signal_count"] + 1,
        )
        logger.info("Signal registered: %s", signal["symbol"])
    else:
        logger.info("No valid setup this cycle")


async def signal_loop():
    """Sleep until next boundary OR wake early when /scan is triggered.
    BUG FIX: we clear the event AFTER wait_for returns successfully, so
    a /scan triggered during a scan is honoured on the next iteration."""
    scan_event: asyncio.Event = AGENT_STATE["scan_trigger"]

    while True:
        now = datetime.now(SLST_TZ)
        day = DAY_NAMES[now.weekday()]

        if in_execution_window():
            try:
                await signal_cycle()
            except Exception as exc:
                logger.error("Signal cycle failed: %s", exc, exc_info=True)
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
            # Event was set → clear it and loop immediately
            scan_event.clear()
            logger.info("🚀 Manual /scan — running now")
        except asyncio.TimeoutError:
            pass


async def signal_loop_supervisor():
    """Restart signal_loop if it crashes completely.
    BUG FIX: previously a signal_loop crash would leave the whole
    trading pipeline dead while other tasks kept running."""
    while True:
        try:
            await signal_loop()
            # signal_loop has `while True`, so this line is unreachable
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Signal loop crashed: %s — restarting in 30s",
                         exc, exc_info=True)
            await asyncio.sleep(30)


async def main():
    update_state(started_at=datetime.now(SLST_TZ))
    AGENT_STATE["scan_trigger"] = asyncio.Event()

    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║  Binance Futures AI Agent — DAY MODE    ║")
    logger.info("╚══════════════════════════════════════════╝")
    logger.info("Model: %s | Reasoning: %s", DEEPSEEK_MODEL, REASONING_EFFORT)
    logger.info("Timeframes: 4H + 1H + 15m | Coins: %d", TOP_COINS_LIMIT)
    logger.info("Candles: 400 | Scan interval: %d min", SCAN_INTERVAL_MIN)
    logger.info("Trading days: Monday – Thursday")
    logger.info("Trading hours: %s – %s SLST",
                SLST_START.strftime("%H:%M"), SLST_END.strftime("%H:%M"))
    logger.info("Off days: Friday, Saturday, Sunday")
    logger.info("Trade monitor runs 24/7 (even outside window)")

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
