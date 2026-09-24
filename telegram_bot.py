"""
Telegram command bot: /help /status /trades /pnl /scan.
Uses Application.builder() + manual start_polling for asyncio.gather.
"""
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from config import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    SLST_START, SLST_END, SLST_UTC_OFFSET,
    SCAN_INTERVAL_MIN, TOP_COINS_LIMIT,
    DEEPSEEK_MODEL, REASONING_EFFORT,
    OFF_WEEKDAYS, DAY_NAMES,
)

logger = logging.getLogger(__name__)
SLST_TZ = timezone(timedelta(hours=SLST_UTC_OFFSET))


AGENT_STATE: Dict[str, Any] = {
    "started_at":   None,
    "last_scan_at": None,
    "scan_count":   0,
    "signal_count": 0,
    "last_signal":  None,
    "scan_trigger": None,
}


def update_state(**kwargs):
    AGENT_STATE.update(kwargs)


def _authorized(update: Update) -> bool:
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
    c = now.replace(hour=SLST_START.hour, minute=SLST_START.minute,
                    second=0, microsecond=0)
    if c <= now or c.weekday() in OFF_WEEKDAYS:
        c += timedelta(days=1)
        while c.weekday() in OFF_WEEKDAYS:
            c += timedelta(days=1)
    return c


def _fmt_dur(seconds: float) -> str:
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"
    return f"{int(seconds // 60)}m"


async def _reply(update: Update, text: str) -> None:
    """Safe reply — handles None update.message."""
    try:
        if update.message:
            await update.message.reply_text(text, parse_mode="Markdown")
        elif update.effective_chat:
            await update.get_bot().send_message(
                chat_id=update.effective_chat.id,
                text=text, parse_mode="Markdown",
            )
    except Exception as exc:
        logger.warning("Reply failed: %s", exc)


# ── Handlers ─────────────────────────────────────────────────────
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return
    await _reply(update,
        "🤖 *Agent Commands*\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "/status — agent health & schedule\n"
        "/trades — active trades + live PnL\n"
        "/pnl    — performance summary\n"
        "/scan   — trigger manual scan now\n"
        "/help   — this message"
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return

    now = _slst_now()
    day = DAY_NAMES[now.weekday()]
    in_win = _in_window(now)

    started = AGENT_STATE.get("started_at")
    uptime = _fmt_dur((now - started).total_seconds()) if started else "?"

    if in_win:
        m = SCAN_INTERVAL_MIN - (now.minute % SCAN_INTERVAL_MIN)
        next_evt = f"next scan in {m}m"
        status_line = "🟢 *ACTIVE* — scanning"
    elif now.weekday() in OFF_WEEKDAYS:
        wait_s = (_next_window_start(now) - now).total_seconds()
        next_evt = f"sleeps until Mon {SLST_START.strftime('%H:%M')} (in {_fmt_dur(wait_s)})"
        status_line = "😴 *OFF-DAY* (Fri/Sat/Sun)"
    else:
        if now.time() < SLST_START:
            wait_s = (now.replace(hour=SLST_START.hour, minute=SLST_START.minute,
                                  second=0, microsecond=0) - now).total_seconds()
        else:
            wait_s = (_next_window_start(now) - now).total_seconds()
        next_evt = f"next window in {_fmt_dur(wait_s)}"
        status_line = "⏸ *OUTSIDE HOURS*"

    ls = AGENT_STATE.get("last_scan_at")
    ls_str = ls.strftime("%H:%M:%S SLST") if ls else "—"
    sig = AGENT_STATE.get("last_signal")
    sig_str = (f"{sig['direction']} {sig['symbol']} ({sig.get('confidence','?')})"
               if sig else "—")

    await _reply(update,
        f"🤖 *Agent Status*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{status_line}\n"
        f"📅 `{day}` | `{now.strftime('%H:%M:%S')} SLST`\n"
        f"⏱ Uptime: `{uptime}`\n"
        f"📊 Scans: `{AGENT_STATE.get('scan_count', 0)}`\n"
        f"🚀 Signals: `{AGENT_STATE.get('signal_count', 0)}`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🔍 Last scan: `{ls_str}`\n"
        f"📈 Last signal: `{sig_str}`\n"
        f"⏭ {next_evt}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⚙️ `{DEEPSEEK_MODEL}` | 🧠 `{REASONING_EFFORT}`\n"
        f"🪙 `{TOP_COINS_LIMIT}` coins | `4H/1H/15m`"
    )


async def cmd_trades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return
    from trade_monitor import get_active_trades, get_prices, calc_pnl

    trades = get_active_trades()
    if not trades:
        await _reply(update, "📭 No active trades.")
        return

    symbols = list({t["symbol"] for t in trades})
    try:
        prices = await get_prices(symbols)
    except Exception:
        prices = {}

    lines = ["📊 *Active Trades*", "━━━━━━━━━━━━━━━━━━━━"]
    for t in trades:
        sym, d = t["symbol"], t["direction"].upper()
        entry = float(t["entry"])
        price = prices.get(sym)
        emoji = "🟢" if d == "LONG" else "🔴"
        pnl_str = f"{calc_pnl(t, price)['pnl_pct']:+.2f}%" if price else "N/A"
        lines.append(
            f"{emoji} `{sym}` *{d}* — `{pnl_str}`\n"
            f"  E: `{entry:.6g}` | SL: `{float(t['sl']):.6g}`\n"
            f"  TP1: `{float(t['tp1']):.6g}` | TP2: `{float(t['tp2']):.6g}` | TP3: `{float(t['tp3']):.6g}`"
        )
    await _reply(update, "\n".join(lines))


async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return
    from trade_monitor import get_closed_trades

    closed = get_closed_trades()
    if not closed:
        await _reply(update, "📭 No closed trades yet.")
        return

    wins = losses = 0
    total_pnl = 0.0
    best = worst = None

    for t in closed:
        e = float(t["entry"]); x = float(t.get("exit_price", e))
        if e == 0:
            continue
        pct = ((x - e) / e * 100) if t["direction"].upper() == "LONG" \
              else ((e - x) / e * 100)
        total_pnl += pct
        wins += 1 if pct >= 0 else 0
        losses += 1 if pct < 0 else 0
        if best is None or pct > best[0]:  best = (pct, t)
        if worst is None or pct < worst[0]: worst = (pct, t)

    total = wins + losses
    wr = (wins / total * 100) if total else 0
    b_str = f"+{best[0]:.2f}% ({best[1]['symbol']})" if best else "—"
    w_str = f"{worst[0]:.2f}% ({worst[1]['symbol']})" if worst else "—"

    await _reply(update,
        f"📈 *Performance Summary*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Total: `{total}` | ✅ `{wins}` | ❌ `{losses}`\n"
        f"Win rate: `{wr:.1f}%`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Cumulative: `{total_pnl:+.2f}%`\n"
        f"Best: `{b_str}`\n"
        f"Worst: `{w_str}`"
    )


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return
    ev = AGENT_STATE.get("scan_trigger")
    if ev is None:
        await _reply(update, "⚠️ Manual scan not available.")
        return
    ev.set()
    await _reply(update, "🚀 Manual scan triggered — running now.")


# ── Bot lifecycle ────────────────────────────────────────────────
async def telegram_bot_loop():
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
    logger.info("Telegram bot started — /help /status /trades /pnl /scan")

    try:
        await asyncio.Event().wait()     # run forever
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
