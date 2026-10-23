"""
v6 ULTRA orchestrator.
60 coins → split exactly into groups of 4 → 15 parallel Nemotron agents.
"""
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone

from config import (
    SLST_START, SLST_END, SLST_UTC_OFFSET,
    TOP_COINS_LIMIT, REASONING_EFFORT,
    OPENROUTER_MODEL, OPENROUTER_API_KEYS,
    SCAN_INTERVAL_MIN, OFF_WEEKDAYS, DAY_NAMES,
)
from data_fetcher import get_top_volume_symbols, fetch_all_data
from indicators import compute_multi_timeframe
from payload_builder import build_payload
from agent import run_agent_batch
from trade_monitor import monitor_loop, add_trade
from telegram_bot import telegram_bot_loop, update_state, AGENT_STATE

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


def _secs_next_window() -> float:
    now = datetime.now(SLST_TZ)
    return max((_next_window_start(now) - now).total_seconds(), 60)


def _secs_next_scan() -> float:
    now = datetime.now(SLST_TZ)
    m = SCAN_INTERVAL_MIN - (now.minute % SCAN_INTERVAL_MIN)
    nxt = (now + timedelta(minutes=m)).replace(second=0, microsecond=0)
    return max((nxt - now).total_seconds(), 30)


def _fmt(s: float) -> str:
    if s >= 3600:
        return f"{int(s // 3600)}h {int((s % 3600) // 60)}m"
    return f"{int(s // 60)}m"


def _split_groups(all_data):
    n_keys = len(OPENROUTER_API_KEYS)
    if n_keys == 0:
        return []

    GROUP_SIZE = 4
    groups = []
    gid = 1
    
    for i in range(0, len(all_data), GROUP_SIZE):
        part = all_data[i:i + GROUP_SIZE]
        if not part:
            continue
        groups.append({
            "id": gid,
            "symbols": [x["symbol"] for x in part],
            "payload": build_payload(part),
            "data": {x["symbol"]: x for x in part},
        })
        gid += 1

    logger.info("Split %d coins into %d groups of %d", len(all_data), len(groups), GROUP_SIZE)
    
    if len(groups) > n_keys:
        logger.warning("Have %d groups but only %d API keys. Extra %d groups will be skipped.", 
                       len(groups), n_keys, len(groups) - n_keys)
    
    return groups


async def signal_cycle():
    logger.info("═══ Signal cycle starting ═══")

    symbols = await get_top_volume_symbols(TOP_COINS_LIMIT)
    bundle = await fetch_all_data(symbols)

    logger.info("Computing indicators …")

    def _compute_all():
        out = []
        for sym, tf_data in bundle.items():
            try:
                out.append(compute_multi_timeframe(sym, tf_data))
            except Exception as exc:
                logger.warning("Indicator error %s: %s", sym, exc)
        return out

    # CPU-heavy → run in a thread so trade monitor / telegram bot keep running
    all_data = await asyncio.to_thread(_compute_all)

    if not all_data:
        logger.warning("No data — skipping")
        return

    groups = _split_groups(all_data)
    if not groups:
        logger.error("No OpenRouter keys configured — cannot analyse")
        return

    best = await run_agent_batch(groups)

    update_state(
        last_scan_at=datetime.now(SLST_TZ),
        scan_count=AGENT_STATE["scan_count"] + 1,
    )

    if best:
        await add_trade(best)
        update_state(
            last_signal=best,
            signal_count=AGENT_STATE["signal_count"] + 1,
        )
        logger.info("Signal registered: %s", best["symbol"])
    else:
        logger.info("No valid setup — discarding iteration")


async def signal_loop():
    ev: asyncio.Event = AGENT_STATE["scan_trigger"]
    while True:
        ev.clear()
        now = datetime.now(SLST_TZ)
        day = DAY_NAMES[now.weekday()]

        if in_execution_window():
            try:
                await signal_cycle()
            except Exception as exc:
                logger.error("Cycle failed: %s", exc, exc_info=True)
            wait = _secs_next_scan()
        else:
            wait = _secs_next_window()
            if now.weekday() in OFF_WEEKDAYS:
                logger.info("[%s] OFF-DAY — sleeping %s", day, _fmt(wait))
            else:
                logger.info("[%s] Outside hours (%s SLST) — sleeping %s",
                            day, now.strftime("%H:%M"), _fmt(wait))

        try:
            await asyncio.wait_for(ev.wait(), timeout=wait)
            logger.info("🚀 Manual /scan — running now")
        except asyncio.TimeoutError:
            pass


async def main():
    update_state(started_at=datetime.now(SLST_TZ))
    AGENT_STATE["scan_trigger"] = asyncio.Event()

    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║  Binance Futures AI Agent — v6 ULTRA    ║")
    logger.info("╚══════════════════════════════════════════╝")
    logger.info("Model: %s", OPENROUTER_MODEL)
    logger.info("Reasoning: %s | API keys: %d",
                REASONING_EFFORT, len(OPENROUTER_API_KEYS))
    logger.info("Timeframes: 1D(100) + 4H(200) + 1H(150) + 15m(100)")
    logger.info("Coins: %d (blacklist excluded)", TOP_COINS_LIMIT)
    logger.info("Groups: %d coins per key",
                max(1, 4))
    logger.info("Scan: every %d min | Mon–Thu", SCAN_INTERVAL_MIN)
    logger.info("Window: %s – %s SLST",
                SLST_START.strftime("%H:%M"), SLST_END.strftime("%H:%M"))

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
