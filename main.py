"""
v3 Main orchestrator — Hybrid vision + text system.
Scan window: 12:00–20:00 SLST (Mon–Thu), every 15 min.
Trade monitor: 24/7, every 5s.
"""
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone

from config import (
    SLST_START, SLST_END, SLST_UTC_OFFSET,
    TOP_COINS_LIMIT, CHART_COIN_LIMIT, CHART_TIMEOUT_SEC,
    REASONING_EFFORT, DEEPSEEK_MODEL,
    SCAN_INTERVAL_MIN, OFF_WEEKDAYS, DAY_NAMES,
)
from data_fetcher import get_top_volume_symbols, fetch_all_data
from indicators import compute_multi_timeframe
from chart_generator import generate_all_charts, chart_manager
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
logging.getLogger("matplotlib").setLevel(logging.WARNING)

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
    cycle_dir = None

    try:
        # 1. Fetch symbols
        symbols = await get_top_volume_symbols(TOP_COINS_LIMIT)
        if not symbols:
            logger.warning("No symbols — skipping cycle")
            return

        # 2. Fetch OHLCV
        logger.info("Fetching OHLCV for %d symbols …", len(symbols))
        bundle = await fetch_all_data(symbols)

        # 3. Compute indicators
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

        # 4. Build text payload
        payload = build_payload(all_data)
        logger.info("Payload size: %.1f KB", len(payload) / 1024)

        # 5. Generate charts (with timeout)
        cycle_dir = chart_manager.new_cycle_dir()
        chart_symbols = symbols[:CHART_COIN_LIMIT]
        try:
            chart_map = await asyncio.wait_for(
                generate_all_charts(bundle, chart_symbols, cycle_dir),
                timeout=CHART_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            logger.error("Chart generation timed out after %ds — "
                         "proceeding with text-only", CHART_TIMEOUT_SEC)
            chart_map = {}

        # 6. Send to DeepSeek
        logger.info("Sending to DeepSeek (model=%s, effort=%s) …",
                    DEEPSEEK_MODEL, REASONING_EFFORT)
        signal = await run_agent(payload, chart_map)

        # 7. Update state
        update_state(
            last_scan_at=datetime.now(SLST_TZ),
            scan_count=AGENT_STATE["scan_count"] + 1,
        )

        # 8. Handle signal
        if signal:
            added = await add_trade(signal)
            if added:
                update_state(
                    last_signal=signal,
                    signal_count=AGENT_STATE["signal_count"] + 1,
                )
                logger.info("Signal registered: %s", signal["symbol"])
            else:
                logger.info("Signal skipped — max positions")
        else:
            logger.info("No valid setup this cycle")

    except Exception as exc:
        logger.error("Signal cycle failed: %s", exc, exc_info=True)

    finally:
        # ALWAYS clean up charts, even if agent crashed
        if cycle_dir:
            chart_manager.cleanup(cycle_dir)


async def signal_loop():
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
            logger.error("Signal loop crashed: %s — restart in 30s", exc)
            await asyncio.sleep(30)


async def main():
    update_state(started_at=datetime.now(SLST_TZ))
    AGENT_STATE["scan_trigger"] = asyncio.Event()

    chart_manager.startup_cleanup()

    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║  Binance Futures AI Agent — v3 HYBRID   ║")
    logger.info("╚══════════════════════════════════════════╝")
    logger.info("Model: %s | Reasoning: %s", DEEPSEEK_MODEL, REASONING_EFFORT)
    logger.info("Timeframes: 4H + 1H + 15m | Coins: %d", TOP_COINS_LIMIT)
    logger.info("Candles: 300 | Charts: %d coins × 3 TFs", CHART_COIN_LIMIT)
    logger.info("Chart size: 1920×1080 @ 100 DPI")
    logger.info("Scan interval: %d min", SCAN_INTERVAL_MIN)
    logger.info("Trading days: Monday – Thursday")
    logger.info("Trading hours: %s – %s SLST",
                SLST_START.strftime("%H:%M"), SLST_END.strftime("%H:%M"))
    logger.info("Off days: Friday, Saturday, Sunday")
    logger.info("Trade monitor: 24/7 (5s interval)")

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
