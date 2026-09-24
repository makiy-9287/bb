"""
Main orchestrator — DAY MODE (Mon–Thu, 12:00–21:00 SLST).

Three concurrent tasks:
  • signal_loop    — 15-min scans, /scan interruptible
  • monitor_loop   — SL/TP tracker
  • telegram_bot   — /status /trades /pnl /scan
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
from telegram_bot import telegram_bot_loop, update_state, AGENT_STATE

# ── logging ──────────────────────────────────────────────────────
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
    now = datetime.now(SLST_TZ)
    m = SCAN_INTERVAL_MIN - (now.minute % SCAN_INTERVAL_MIN)
    nxt = (now + timedelta(minutes=m)).replace(second=0, microsecond=0)
    return max((nxt - now).total_seconds(), 30)


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
        logger.warning("No data computed — skipping cycle")
        return

    payload = build_payload(all_data)
    logger.info("Payload size: %.1f KB", len(payload) / 1024)

    logger.info("Sending payload (model=%s, effort=%s) …",
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
        logger.info("No valid setup — discarding iteration")


async def signal_loop():
    """15-min scans during window; interruptible by /scan."""
    scan_event: asyncio.Event = AGENT_STATE["scan_trigger"]

    while True:
        scan_event.clear()
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

        # Sleep OR wake early on /scan
        try:
            await asyncio.wait_for(scan_event.wait(), timeout=wait_sec)
            logger.info("🚀 Manual /scan — running now")
        except asyncio.TimeoutError:
            pass


async def main():
    update_state(started_at=datetime.now(SLST_TZ))
    AGENT_STATE["scan_trigger"] = asyncio.Event()

    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║  Binance Futures AI Agent — DAY MODE    ║")
    logger.info("╚══════════════════════════════════════════╝")
    logger.info("Model: %s | Reasoning: %s", DEEPSEEK_MODEL, REASONING_EFFORT)
    logger.info("Timeframes: 4H + 1H + 15m | Coins: %d", TOP_COINS_LIMIT)
    logger.info("Candles: 300 | Scan interval: %d min", SCAN_INTERVAL_MIN)
    logger.info("Trading days: Monday – Thursday")
    logger.info("Trading hours: %s – %s SLST",
                SLST_START.strftime("%H:%M"), SLST_END.strftime("%H:%M"))
    logger.info("Off days: Friday, Saturday, Sunday")

    results = await asyncio.gather(
        signal_loop(),
        monitor_loop(),
        telegram_bot_loop(),
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
