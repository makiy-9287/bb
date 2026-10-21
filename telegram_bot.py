"""
Telegram command bot — /start /help /status /trades /pnl /scan.
"""
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.request import HTTPXRequest

from config import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    SLST_START, SLST_END, SLST_UTC_OFFSET,
    SCAN_INTERVAL_MIN, TOP_COINS_LIMIT,
    OPENROUTER_MODEL, REASONING_EFFORT,  # 👈 මෙතන වෙනස් කළා
    OFF_WEEKDAYS, DAY_NAMES,
)

logger = logging.getLogger(__name__)
SLST_TZ = timezone(timedelta(hours=SLST_UTC_OFFSET))

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("telegram.ext.Updater").setLevel(logging.WARNING)


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


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    if not chat:
        return
    if _authorized(update):
        await _reply(update,
            "🤖 *Agent Commands*\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "/status — agent health\n"
            "/trades — active trades\n"
            "/pnl    — performance summary\n"
            "/scan   — trigger manual scan\n"
            "/help   — this message"
        )
    else:
        await _reply(update,
            f"👋 *Welcome!*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Your Telegram Chat ID:\n`{chat.id}`\n\n"
            f"Send this to the bot admin."
        )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        await cmd_start(update, context)
        return
    await _reply(update,
        "🤖 *Agent Commands*\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "/status /trades /pnl /scan /help"
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return
    now = _slst_now()
    day = DAY_NAMES[now.weekday()]
    started = AGENT_STATE.get("started_at")
    uptime = _fmt_dur((now - started).total_seconds()) if started else "?"

    if _in_window(now):
        m = SCAN_INTERVAL_MIN - (now.minute % SCAN_INTERVAL_MIN)
        next_evt = f"next scan in {m}m"
        status_line = "🟢 *ACTIVE*"
    elif now.weekday() in OFF_WEEKDAYS:
        wait_s = (_next_window_start(now) - now).total_seconds()
        next_evt = f"next Mon in {_fmt_dur(wait_s)}"
        status_line = "😴 *OFF-DAY*"
    else:
        wait_s = (_next_window_start(now) - now).total_seconds()
        next_evt = f"next window in {_fmt_dur(wait_s)}"
        status_line = "⏸ *OUTSIDE HOURS*"

    ls = AGENT_STATE.get("last_scan_at")
    ls_str = ls.strftime("%H:%M:%S SLST") if ls else "—"
    sig = AGENT_STATE.get("last_signal")
    sig_str = (f"{sig['direction']} {sig['symbol']} "
               f"({sig.get('confidence','?')})" if sig else "—")

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
        f"⚙️ `{OPENROUTER_MODEL}` | 🧠 `{REASONING_EFFORT}`\n" # 👈 මෙතනත් වෙනස් කළා
        f"🪙 `{TOP_COINS_LIMIT}` coins | `1D/4H/1H/15m`" # 👈 1D එකතු කළා
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
        tps = []
        if t.get("tp1_hit"): tps.append("TP1✓")
        if t.get("tp2_hit"): tps.append("TP2✓")
        tps_mark = f" [{','.join(tps)}]" if tps else ""
        lines.append(
            f"{emoji} `{sym}` *{d}* — `{pnl_str}`{tps_mark}\n"
            f"  E: `{entry:.6g}` | SL: `{float(t['sl']):.6g}`\n"
            f"  TP1: `{float(t['tp1']):.6g}` | "
            f"TP2: `{float(t['tp2']):.6g}` | "
            f"TP3: `{float(t['tp3']):.6g}`"
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
    total = 0.0
    best = worst = None
    for t in closed:
        e = float(t["entry"])
        x = float(t.get("exit_price", e))
        if e == 0:
            continue
        pct = ((x - e) / e * 100) if t["direction"].upper() == "LONG" \
              else ((e - x) / e * 100)
        total += pct
        wins += 1 if pct >= 0 else 0
        losses += 1 if pct < 0 else 0
        if best is None or pct > best[0]:  best = (pct, t)
        if worst is None or pct < worst[0]: worst = (pct, t)
    n = wins + losses
    wr = (wins / n * 100) if n else 0
    b_str = f"+{best[0]:.2f}% ({best[1]['symbol']})" if best else "—"
    w_str = f"{worst[0]:.2f}% ({worst[1]['symbol']})" if worst else "—"
    await _reply(update,
        f"📈 *Performance Summary*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Total: `{n}` | ✅ `{wins}` | ❌ `{losses}`\n"
        f"Win rate: `{wr:.1f}%`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Cumulative: `{total:+.2f}%`\n"
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
    await _reply(update, "🚀 Manual scan triggered.")


def _on_polling_error(exc: Exception) -> None:
    """MUST be a regular function (not async) per python-telegram-bot v21."""
    logger.warning("Telegram polling hiccup: %s — retrying", type(exc).__name__)


async def telegram_bot_loop():
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN not set — bot disabled")
        return

    request = HTTPXRequest(
        connection_pool_size=8,
        connect_timeout=30.0,
        read_timeout=60.0,
        write_timeout=30.0,
        pool_timeout=30.0,
    )
    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .request(request)
        .get_updates_request(request)
        .build()
    )
    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("help",   cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("trades", cmd_trades))
    app.add_handler(CommandHandler("pnl",    cmd_pnl))
    app.add_handler(CommandHandler("scan",   cmd_scan))

    await app.initialize()
    await app.start()
    await app.updater.start_polling(
        drop_pending_updates=True,
        error_callback=_on_polling_error,
    )
    logger.info("Telegram bot started — /start /help /status /trades /pnl /scan")

    try:
        await asyncio.Event().wait()
    finally:
        try:
            await app.updater.stop()
            await app.stop()
            await app.shutdown()
        except Exception as exc:
            logger.warning("Shutdown error: %s", exc)


async def telegram_bot_supervisor():
    while True:
        try:
            await telegram_bot_loop()
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Bot crashed: %s — restart in 30s", exc)
            await asyncio.sleep(30)
