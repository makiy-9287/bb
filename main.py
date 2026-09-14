"""
Main orchestrator.  Runs the hourly signal loop (within the SLST
execution window) and the continuous trade monitor concurrently.
"""
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone

from config import (
    SLST_START, SLST_END, SLST_UTC_OFFSET,
    TOP_COINS_LIMIT, REASONING_EFFORT, DEEPSEEK_MODEL
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
logger = logging.getLogger("main")

SLST_TZ = timezone(timedelta(hours=SLST_UTC_OFFSET))


def in_execution_window() -> bool:
    """Return True if current SLST time is within [05:00, 21:00)."""
    now_slst = datetime.now(SLST_TZ)
    t = now_slst.time()
    return SLST_START <= t < SLST_END


def _seconds_until_next_window() -> float:
    """If outside the window, compute seconds until the next 05:00 SLST."""
    now_slst = datetime.now(SLST_TZ)
    target = now_slst.replace(
        hour=SLST_START.hour, minute=SLST_START.minute,
        second=0, microsecond=0
    )
    if now_slst >= target:
        target += timedelta(days=1)
    return (target - now_slst).total_seconds()


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
    """Run signal_cycle every hour while inside the execution window."""
    while True:
        if in_execution_window():
            try:
                await signal_cycle()
            except Exception as exc:
                logger.error("Signal cycle failed: %s", exc, exc_info=True)

            # Sleep to the next hour boundary (guard against negative)
            now = datetime.now(SLST_TZ)
            next_hour = (now.replace(minute=0, second=0, microsecond=0)
                         + timedelta(hours=1))
            wait_sec = max((next_hour - now).total_seconds(), 30)
        else:
            wait_sec = max(_seconds_until_next_window(), 60)
            logger.info("Outside execution window — sleeping %.0f min",
                        wait_sec / 60)

        await asyncio.sleep(wait_sec)


async def main():
    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║  Binance Futures AI Signal Agent v1.1   ║")
    logger.info("╚══════════════════════════════════════════╝")
    logger.info("Model: %s | Reasoning effort: %s",
                DEEPSEEK_MODEL, REASONING_EFFORT)
    logger.info("Execution window: %s – %s SLST",
                SLST_START.strftime("%H:%M"), SLST_END.strftime("%H:%M"))

    await asyncio.gather(
        signal_loop(),
        monitor_loop(),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully …")