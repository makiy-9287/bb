"""
Main orchestrator — DAY TRADING MODE.

Trading days: Monday – Thursday only.
Trading hours: 12:00 – 21:00 SLST.
Friday, Saturday, Sunday: agent sleeps (no scans).

Scan loop runs every 15 minutes inside the execution window.
Trade monitor runs continuously in parallel (also paused on off-days).
"""
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone

from config import (
    SLST_START, SLST_END, SLST_UTC_OFFSET,
    TOP_COINS_LIMIT, REASONING_EFFORT, DEEPSEEK_MODEL,
    SCAN_INTERVAL_MIN,
)
from data_fetcher import get_top_volume_symbols, fetch_all_data
from indicators import compute_multi_timeframe
from payload_builder import build_payload
from agent import run_agent
from trade_monitor import monitor_loop, add_trade

# ── logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("agent.log"),
    ],
)
logging.getLogger("pandas_ta_classic").setLevel(logging.ERROR)
logging.getLogger("pandas_ta_classic.utils.core").setLevel(logging.ERROR)

logger = logging.getLogger("main")
SLST_TZ = timezone(timedelta(hours=SLST_UTC_OFFSET))

# ── Trading day config ───────────────────────────────────────────
# Python weekday(): Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6
# We only trade Monday–Thursday.
TRADING_WEEKDAYS = {0, 1, 2, 3}   # Mon, Tue, Wed, Thu
OFF_WEEKDAYS     = {4, 5, 6}      # Fri, Sat, Sun
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def in_execution_window() -> bool:
    """
    Return True if:
      • Current SLST day is Monday–Thursday (skip Fri/Sat/Sun)
      • Current SLST time is within [12:00, 21:00)
    """
    now_slst = datetime.now(SLST_TZ)

    # Skip Friday, Saturday, Sunday
    if now_slst.weekday() in OFF_WEEKDAYS:
        return False

    t = now_slst.time()
    return SLST_START <= t < SLST_END


def _next_window_start(now_slst: datetime) -> datetime:
    """Find the next valid trading-window start:
    the next Mon–Thu day at SLST_START (12:00 SLST)."""
    candidate = now_slst.replace(
        hour=SLST_START.hour, minute=SLST_START.minute,
        second=0, microsecond=0,
    )

    # If today's window start has already passed,
    # OR today is Fri/Sat/Sun, advance to the next trading day.
    if candidate <= now_slst or candidate.weekday() in OFF_WEEKDAYS:
        candidate += timedelta(days=1)
        while candidate.weekday() in OFF_WEEKDAYS:
            candidate += timedelta(days=1)

    return candidate


def _seconds_until_next_window() -> float:
    """Seconds until the next valid trading window opens."""
    now_slst = datetime.now(SLST_TZ)
    target = _next_window_start(now_slst)
    return max((target - now_slst).total_seconds(), 60)


def _seconds_to_next_scan_boundary() -> float:
    """Seconds until the next 15-minute boundary."""
    now = datetime.now(SLST_TZ)
    minutes_to_add = SCAN_INTERVAL_MIN - (now.minute % SCAN_INTERVAL_MIN)
    next_boundary = (now + timedelta(minutes=minutes_to_add)).replace(
        second=0, microsecond=0
    )
    return max((next_boundary - now).total_seconds(), 30)


def _fmt_sleep(seconds: float) -> str:
    """Pretty-format seconds into 'X h Y m' or 'Y m'."""
    if seconds >= 3600:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"
    return f"{int(seconds // 60)}m"


async def signal_cycle():
    """One full iteration: fetch → compute → payload → agent."""
    logger.info("═══ Signal cycle starting ═══")

    symbols = await get_top_volume_symbols(TOP_COINS_LIMIT)
    logger.info("Fetching OHLCV for %d symbols …", len(symbols))
    bundle = await fetch_all_data(symbols)

    logger.info("Computing indicators …")
    all_data = []
    for sym, tf_data in bundle.items():
        try:
            computed = compute_multi_timeframe(sym, tf_data)
            all_data.append(computed)
        except Exception as exc:
            logger.warning("Indicator error %s: %s", sym, exc)

    if not all_data:
        logger.warning("No data computed — skipping cycle")
        return

    payload = build_payload(all_data)
    logger.info("Payload size: %.1f KB", len(payload) / 1024)

    logger.info("Sending payload to DeepSeek agent (model=%s, effort=%s) …",
                DEEPSEEK_MODEL, REASONING_EFFORT)
    signal = await run_agent(payload)

    if signal:
        await add_trade(signal)
        logger.info("Signal registered for monitoring: %s", signal["symbol"])
    else:
        logger.info("No valid setup this cycle — discarding iteration")


async def signal_loop():
    """Scan every SCAN_INTERVAL_MIN minutes, Mon–Thu, 12:00–21:00 SLST."""
    while True:
        now = datetime.now(SLST_TZ)
        day_name = DAY_NAMES[now.weekday()]

        if in_execution_window():
            try:
                await signal_cycle()
            except Exception as exc:
                logger.error("Signal cycle failed: %s", exc, exc_info=True)

            wait_sec = _seconds_to_next_scan_boundary()
        else:
            wait_sec = _seconds_until_next_window()

            # Clear, informative log message
            if now.weekday() in OFF_WEEKDAYS:
                logger.info("[%s] Market OFF-DAY — agent sleeping %s until next window",
                            day_name, _fmt_sleep(wait_sec))
            else:
                logger.info("[%s] Outside trading hours (%s SLST) — sleeping %s",
                            day_name, now.strftime("%H:%M"), _fmt_sleep(wait_sec))

        await asyncio.sleep(wait_sec)


async def main():
    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║  Binance Futures AI Agent — DAY MODE    ║")
    logger.info("╚══════════════════════════════════════════╝")
    logger.info("Model: %s | Reasoning effort: %s",
                DEEPSEEK_MODEL, REASONING_EFFORT)
    logger.info("Timeframes: 4H + 1H + 15m | Coins: %d", TOP_COINS_LIMIT)
    logger.info("Candles per timeframe: 300")
    logger.info("Scan interval: %d min", SCAN_INTERVAL_MIN)
    logger.info("Trading days: Monday – Thursday")
    logger.info("Trading hours: %s – %s SLST (Sri Lanka Time)",
                SLST_START.strftime("%H:%M"), SLST_END.strftime("%H:%M"))
    logger.info("Off days: Friday, Saturday, Sunday")

    await asyncio.gather(
        signal_loop(),
        monitor_loop(),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully …")
