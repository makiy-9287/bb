"""
v3.1 Main orchestrator — Batched hybrid vision + text system.

Each scan cycle:
  1. Fetch 40 coins' OHLCV
  2. Compute indicators (once)
  3. Split into N batches of BATCH_SIZE
  4. For each batch: build payload → generate charts → send to agent
  5. Pick best signal across all batches → register trade

Scan window: 12:00–20:00 SLST (Mon–Thu), every 15 min.
Trade monitor: 24/7, every 5s.
"""
import asyncio
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from config import (
    SLST_START, SLST_END, SLST_UTC_OFFSET,
    TOP_COINS_LIMIT, CHART_COIN_LIMIT, CHART_TIMEOUT_SEC,
    REASONING_EFFORT, DEEPSEEK_MODEL, BATCH_SIZE,
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

_CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}


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


# ── Signal selection ─────────────────────────────────────────────
def _signal_score(sig: Dict[str, Any]) -> tuple:
    """Rank signals: primary = confidence, secondary = R:R to TP3."""
    conf = _CONFIDENCE_RANK.get(sig.get("confidence", "low"), 1)
    try:
        entry = float(sig["entry"])
        sl = float(sig["sl"])
        tp3 = float(sig["tp3"])
        risk = abs(entry - sl)
        reward = abs(tp3 - entry)
        rr = reward / risk if risk > 0 else 0.0
    except Exception:
        rr = 0.0
    return (conf, rr)


def _pick_best_signal(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
    return max(signals, key=_signal_score)


# ── Batch processor ──────────────────────────────────────────────
async def process_batch(
    batch_num: int,
    total_batches: int,
    batch_symbols: List[str],
    bundle: Dict[str, Dict],
    all_data: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Process one batch end-to-end. Always cleans up its own charts."""
    tag = f"b{batch_num}"
    prefix = f"[Batch {batch_num}/{total_batches}]"
    t0 = time.time()

    logger.info("%s ━━━ START — %d coins ━━━", prefix, len(batch_symbols))
    logger.info("%s Coins: %s", prefix, ", ".join(batch_symbols))

    batch_dir = chart_manager.new_cycle_dir(tag)
    logger.info("%s Chart dir: %s", prefix, batch_dir)

    try:
        # 1. Filter computed data for this batch
        logger.info("%s Filtering indicator data …", prefix)
        batch_data = [d for d in all_data if d["symbol"] in batch_symbols]
        logger.info("%s Filtered → %d/%d symbols have data",
                    prefix, len(batch_data), len(batch_symbols))

        if not batch_data:
            logger.warning("%s No indicator data — skipping batch", prefix)
            return None

        # 2. Build text payload
        logger.info("%s Building text payload …", prefix)
        payload = build_payload(batch_data)
        logger.info("%s Text payload: %.1f KB (%d chars)",
                    prefix, len(payload) / 1024, len(payload))

        # 3. Generate charts
        logger.info("%s Generating charts for %d coins …",
                    prefix, len(batch_symbols))
        t_charts = time.time()
        try:
            chart_map = await asyncio.wait_for(
                generate_all_charts(bundle, batch_symbols, batch_dir),
                timeout=CHART_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            logger.error("%s Chart generation timed out after %ds — "
                         "proceeding with partial/text-only",
                         prefix, CHART_TIMEOUT_SEC)
            chart_map = {}

        total_charts = sum(len(v) for v in chart_map.values())
        logger.info("%s Charts ready: %d in %.1fs",
                    prefix, total_charts, time.time() - t_charts)

        # 4. Send to DeepSeek
        header = (f"=== BATCH {batch_num}/{total_batches} — "
                  f"{len(batch_data)} coins ===\n\n")
        logger.info("%s Sending to DeepSeek (model=%s, effort=%s) …",
                    prefix, DEEPSEEK_MODEL, REASONING_EFFORT)
        t_api = time.time()
        signal = await run_agent(header + payload, chart_map)
        logger.info("%s Agent response in %.1fs",
                    prefix, time.time() - t_api)

        # 5. Report result
        if signal:
            logger.info("%s ✅ Signal: %s %s (confidence=%s)",
                        prefix, signal["direction"], signal["symbol"],
                        signal.get("confidence"))
        else:
            logger.info("%s ✗ No valid setup in this batch", prefix)

        logger.info("%s ━━━ END in %.1fs ━━━", prefix, time.time() - t0)
        return signal

    except Exception as exc:
        logger.error("%s Crashed: %s", prefix, exc, exc_info=True)
        return None

    finally:
        logger.info("%s Cleaning up charts: %s", prefix, batch_dir)
        chart_manager.cleanup(batch_dir)


# ── Signal cycle ─────────────────────────────────────────────────
async def signal_cycle():
    cycle_start = time.time()
    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║  ═══ Signal cycle starting ═══          ║")
    logger.info("╚══════════════════════════════════════════╝")

    try:
        # 1. Symbols
        logger.info("Step 1/5 — Fetching top symbols by volume …")
        symbols = await get_top_volume_symbols(TOP_COINS_LIMIT)
        if not symbols:
            logger.warning("No symbols returned — cycle aborted")
            return
        logger.info("Step 1/5 — Done: %d symbols", len(symbols))

        # 2. OHLCV
        logger.info("Step 2/5 — Fetching OHLCV (4H/1H/15m) …")
        t = time.time()
        bundle = await fetch_all_data(symbols)
        logger.info("Step 2/5 — Done: %d symbols in %.1fs",
                    len(bundle), time.time() - t)

        # 3. Indicators
        logger.info("Step 3/5 — Computing indicators …")
        t = time.time()
        all_data: List[Dict[str, Any]] = []
        for sym, tf_data in bundle.items():
            try:
                all_data.append(compute_multi_timeframe(sym, tf_data))
            except Exception as exc:
                logger.warning("Indicator error %s: %s", sym, exc)
        logger.info("Step 3/5 — Done: %d symbols in %.1fs",
                    len(all_data), time.time() - t)

        if not all_data:
            logger.warning("No indicator data — cycle aborted")
            return

        # 4. Batches
        batches = [symbols[i:i + BATCH_SIZE]
                   for i in range(0, len(symbols), BATCH_SIZE)]
        logger.info("Step 4/5 — Split into %d batches of ≤%d coins",
                    len(batches), BATCH_SIZE)
        for i, b in enumerate(batches, 1):
            logger.info("  • Batch %d: %d coins — %s",
                        i, len(b), ", ".join(b[:5]) +
                        (" …" if len(b) > 5 else ""))

        # 5. Process batches sequentially
        logger.info("Step 5/5 — Processing batches sequentially …")
        signals: List[Dict[str, Any]] = []
        for i, batch in enumerate(batches, 1):
            sig = await process_batch(
                i, len(batches), batch, bundle, all_data,
            )
            if sig:
                signals.append(sig)

        # 6. Pick best
        logger.info("━━━ Cycle summary ━━━")
        logger.info("Batches processed: %d", len(batches))
        logger.info("Signals collected: %d", len(signals))

        update_state(
            last_scan_at=datetime.now(SLST_TZ),
            scan_count=AGENT_STATE["scan_count"] + 1,
        )

        if signals:
            best = _pick_best_signal(signals)
            logger.info("🏆 Best signal: %s %s (confidence=%s)",
                        best["direction"], best["symbol"],
                        best.get("confidence"))

            added = await add_trade(best)
            if added:
                update_state(
                    last_signal=best,
                    signal_count=AGENT_STATE["signal_count"] + 1,
                )
                logger.info("✓ Trade registered: %s", best["symbol"])
            else:
                logger.info("✗ Trade skipped — max positions reached")
        else:
            logger.info("✗ No valid setup from any batch this cycle")

    except Exception as exc:
        logger.error("Signal cycle crashed: %s", exc, exc_info=True)

    finally:
        elapsed = time.time() - cycle_start
        logger.info("═══ Cycle complete in %.1fs ═══\n", elapsed)


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

    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║  Binance Futures AI Agent — v3.1 BATCHED    ║")
    logger.info("╚══════════════════════════════════════════════╝")
    logger.info("Model: %s | Reasoning: %s", DEEPSEEK_MODEL, REASONING_EFFORT)
    logger.info("Timeframes: 4H + 1H + 15m")
    logger.info("Coins: %d | Batch size: %d (%d batches/cycle)",
                TOP_COINS_LIMIT, BATCH_SIZE,
                (TOP_COINS_LIMIT + BATCH_SIZE - 1) // BATCH_SIZE)
    logger.info("Candles: 300 | Chart: 1920×1080 @ 100 DPI")
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
