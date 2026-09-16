"""
Telegram bot with slash commands for monitoring the agent.

Commands:
    /help   — list all commands
    /status — agent health, schedule, next scan time
    /trades — active trades with live PnL
    /pnl    — overall performance summary (from closed trades)
    /scan   — trigger a manual scan immediately (uses current market data)
"""
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from config import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    SLST_START, SLST_END, SLST_UTC_OFFSET,
    SCAN_INTERVAL_MIN, TOP_COINS_LIMIT,
    DEEPSEEK_MODEL, REASONING_EFFORT,
    TRADING_WEEKDAYS, OFF_WEEKDAYS, DAY_NAMES,
)

logger = logging.getLogger(__name__)
SLST_TZ = timezone(timedelta(hours=SLST_UTC_OFFSET))


# ── Shared state (updated by main.py) ────────────────────────────
AGENT_STATE: Dict[str, Any] = {
    "started_at":   None,
    "last_scan_at": None,
    "scan_count":   0,
    "signal_count": 0,
    "last_signal":  None,
    "scan_trigger": None,   # asyncio.Event for /scan command
}


def update_state(**kwargs):
    """Update shared agent state (called from main.py)."""
    AGENT_STATE.update(kwargs)


# ── Auth guard ───────────────────────────────────────────────────
def _authorized(update: Update) -> bool:
    """Only respond to the configured chat ID."""
    if not update.effective_chat:
        return False
    return str(update.effective_chat.id) == str(TELEGRAM_CHAT_ID)


def _slst_now() -> datetime:
    return datetime.now(SLST_TZ)


def _in_window(now: datetime | None = None) -> bool:
    now = now or _slst_now()
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


def _fmt_duration(seconds: float) -> str:
    if seconds >= 3600:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"
    return f"{int(seconds // 60)}m"


# ── Command handlers ─────────────────────────────────────────────
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return
    text = (
        "🤖 *Agent Commands*\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "/status — agent health & schedule\n"
        "/trades — active trades + live PnL\n"
        "/pnl    — overall performance summary\n"
        "/scan   — trigger manual scan now\n"
        "/help   — this message"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return

    now = _slst_now()
    day = DAY_NAMES[now.weekday()]
    in_win = _in_window(now)

    # Uptime
    started = AGENT_STATE.get("started_at")
    if started:
        uptime = _fmt_duration((now - started).total_seconds())
    else:
        uptime = "?"

    # Next scan / next window
    if in_win:
        # minutes to next 15-min boundary
        m = SCAN_INTERVAL_MIN - (now.minute % SCAN_INTERVAL_MIN)
        next_evt = f"next scan in {m}m"
        status_line = "🟢 *ACTIVE* — scanning"
    elif now.weekday() in OFF_WEEKDAYS:
        wait_s = (_next_window_start(now) - now).total_seconds()
        next_evt = f"sleeps until Monday {SLST_START.strftime('%H:%M')} (in {_fmt_duration(wait_s)})"
        status_line = "😴 *OFF-DAY* (Fri/Sat/Sun)"
    else:
        if now.time() < SLST_START:
            wait_s = (now.replace(hour=SLST_START.hour, minute=SLST_START.minute,
                                  second=0, microsecond=0) - now).total_seconds()
        else:
            wait_s = (_next_window_start(now) - now).total_seconds()
        next_evt = f"next window in {_fmt_duration(wait_s)}"
        status_line = "⏸ *OUTSIDE HOURS*"

    last_scan = AGENT_STATE.get("last_scan_at")
    last_scan_str = last_scan.strftime("%H:%M:%S SLST") if last_scan else "—"

    last_sig = AGENT_STATE.get("last_signal")
    if last_sig:
        last_sig_str = (f"{last_sig['direction']} {last_sig['symbol']} "
                        f"({last_sig.get('confidence','?')})")
    else:
        last_sig_str = "—"

    text = (
        f"🤖 *Agent Status*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{status_line}\n"
        f"📅 Day: `{day}` | Time: `{now.strftime('%H:%M:%S')} SLST`\n"
        f"⏱ Uptime: `{uptime}`\n"
        f"📊 Scans run: `{AGENT_STATE.get('scan_count', 0)}`\n"
        f"🚀 Signals sent: `{AGENT_STATE.get('signal_count', 0)}`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🔍 Last scan: `{last_scan_str}`\n"
        f"📈 Last signal: `{last_sig_str}`\n"
        f"⏭ {next_evt}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⚙️ Model: `{DEEPSEEK_MODEL}`\n"
        f"🧠 Reasoning: `{REASONING_EFFORT}`\n"
        f"🪙 Coins: `{TOP_COINS_LIMIT}` | TFs: `4H/1H/15m`"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_trades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return

    # Imported here to avoid circular import
    from trade_monitor import get_active_trades, get_prices, calc_pnl

    trades = get_active_trades()
    if not trades:
        await update.message.reply_text("📭 No active trades.")
        return

    symbols = list({t["symbol"] for t in trades})
    try:
        prices = await get_prices(symbols)
    except Exception as exc:
        logger.error("Failed to fetch prices: %s", exc)
        prices = {}

    lines = ["📊 *Active Trades*", "━━━━━━━━━━━━━━━━━━━━"]
    for i, t in enumerate(trades, 1):
        sym = t["symbol"]
        direction = t["direction"].upper()
        entry = float(t["entry"])
        price = prices.get(sym)
        emoji = "🟢" if direction == "LONG" else "🔴"

        if price is None:
            pnl_str = "price N/A"
        else:
            pnl = calc_pnl(t, price)
            sign = "+" if pnl["pnl_pct"] >= 0 else ""
            pnl_str = f"{sign}{pnl['pnl_pct']:.2f}%"

        lines.append(
            f"{emoji} `{sym}` *{direction}* — PnL: `{pnl_str}`\n"
            f"   E: `{entry:.6g}` | SL: `{float(t['sl']):.6g}`\n"
            f"   TP1: `{float(t['tp1']):.6g}` | TP2: `{float(t['tp2']):.6g}` | TP3: `{float(t['tp3']):.6g}`\n"
            f"   Opened: `{t.get('opened_at','?')[:16].replace('T',' ')}`\n"
        )

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return

    from trade_monitor import get_closed_trades

    closed = get_closed_trades()
    if not closed:
        await update.message.reply_text("📭 No closed trades yet.")
        return

    wins = losses = 0
    total_pnl_pct = 0.0
    best = worst = None

    for t in closed:
        e = float(t["entry"])
        x = float(t.get("exit_price", e))
        d = t["direction"].upper()
        if e == 0:
            continue
        pnl_pct = ((x - e) / e * 100) if d == "LONG" else ((e - x) / e * 100)
        total_pnl_pct += pnl_pct
        if pnl_pct >= 0:
            wins += 1
        else:
            losses += 1
        if best is None or pnl_pct > best[0]:
            best = (pnl_pct, t)
        if worst is None or pnl_pct < worst[0]:
            worst = (pnl_pct, t)

    total = wins + losses
    winrate = (wins / total * 100) if total else 0.0

    best_str = f"+{best[0]:.2f}% ({best[1]['symbol']})" if best else "—"
    worst_str = f"{worst[0]:.2f}% ({worst[1]['symbol']})" if worst else "—"

    text = (
        f"📈 *Performance Summary*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Total closed: `{total}`\n"
        f"Wins: `{wins}` ✅ | Losses: `{losses}` ❌\n"
        f"Win rate: `{winrate:.1f}%`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Cumulative PnL: `{total_pnl_pct:+.2f}%`\n"
        f"Best: `{best_str}`\n"
        f"Worst: `{worst_str}`"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return
    ev = AGENT_STATE.get("scan_trigger")
    if ev is None:
        await update.message.reply_text("⚠️ Manual scan not available.")
        return
    ev.set()
    await update.message.reply_text("🚀 Manual scan triggered — check logs.")


# ── Bot entry point ──────────────────────────────────────────────
async def telegram_bot_loop():
    """Run the Telegram bot indefinitely. Cancellable."""
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN not set — bot disabled")
        return

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("help",   cmd_help))
    app.add_handler(CommandHandler("start",  cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("trades", cmd_trades))
    app.add_handler(CommandHandler("pnl",    cmd_pnl))
    app.add_handler(CommandHandler("scan",   cmd_scan))

    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)
    logger.info("Telegram bot started (commands: /help /status /trades /pnl /scan)")

    try:
        await asyncio.Event().wait()   # run forever
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
