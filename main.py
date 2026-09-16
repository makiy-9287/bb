"""
Main orchestrator — DAY TRADING MODE (Mon–Thu, 12:00–21:00 SLST).

Runs three concurrent async tasks:
    1. signal_loop    — scans every 15 min during the window
    2. monitor_loop   — tracks SL/TP of active trades continuously
    3. telegram_bot   — listens for /status, /trades, /pnl commands
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
from telegram_bot import (
    telegram_bot_loop, update_state, AGENT_STATE,
)

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
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger("main")
SLST_TZ = timezone(timedelta(hours=SLST_UTC_OFFSET))


# ── Window logic ─────────────────────────────────────────────────
def in_execution_window() -> bool:
    now = datetime.now(SLST_TZ)
    if now.weekday() in OFF_WEEKDAYS:
        return False
    return SLST_START <= now.time() < SLST_END


def _next_window_start(now: datetime) -> datetime:
    candidate = now.replace(
        hour=SLST_START.hour, minute=SLST_START.minute,
        second=0, microsecond=0,
    )
    if candidate <= now or candidate.weekday() in OFF_WEEKDAYS:
        candidate += timedelta(days=1)
        while candidate.weekday() in OFF_WEEKDAYS:
            candidate += timedelta(days=1)
    return candidate


def _seconds_until_next_window() -> float:
    now = datetime.now(SLST_TZ)
    return max((_next_window_start(now) - now).total_seconds(), 60)


def _seconds_to_next_scan_boundary() -> float:
    now = datetime.now(SLST_TZ)
    m = SCAN_INTERVAL_MIN - (now.minute % SCAN_INTERVAL_MIN)
    nxt = (now + timedelta(minutes=m)).replace(second=0, microsecond=0)
    return max((nxt - now).total_seconds(), 30)


def _fmt_sleep(seconds: float) -> str:
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"
    return f"{int(seconds // 60)}m"


# ── Signal cycle ─────────────────────────────────────────────────
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

    logger.info("Sending payload to DeepSeek (model=%s, effort=%s) …",
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
        logger.info("Signal registered for monitoring: %s", signal["symbol"])
    else:
        logger.info("No valid setup this cycle — discarding iteration")


# ── Signal loop ──────────────────────────────────────────────────
async def signal_loop():
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
                logger.info("[%s] Market OFF-DAY — sleeping %s",
                            day, _fmt_sleep(wait_sec))
            else:
                logger.info("[%s] Outside hours (%s SLST) — sleeping %s",
                            day, now.strftime("%H:%M"), _fmt_sleep(wait_sec))

        await asyncio.sleep(wait_sec)


# ── Main ─────────────────────────────────────────────────────────
async def main():
    update_state(started_at=datetime.now(SLST_TZ))

    # Shared event so /scan command can trigger manual scans
    AGENT_STATE["scan_trigger"] = asyncio.Event()

    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║  Binance Futures AI Agent — DAY MODE    ║")
    logger.info("╚══════════════════════════════════════════╝")
    logger.info("Model: %s | Reasoning effort: %s",
                DEEPSEEK_MODEL, REASONING_EFFORT)
    logger.info("Timeframes: 4H + 1H + 15m | Coins: %d", TOP_COINS_LIMIT)
    logger.info("Candles per timeframe: 300")
    logger.info("Scan interval: %d min", SCAN_INTERVAL_MIN)
    logger.info("Trading days: Monday – Thursday")
    logger.info("Trading hours: %s – %s SLST",
                SLST_START.strftime("%H:%M"), SLST_END.strftime("%H:%M"))
    logger.info("Off days: Friday, Saturday, Sunday")

    await asyncio.gather(
        signal_loop(),
        monitor_loop(),
        telegram_bot_loop(),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully …")
