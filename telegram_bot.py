"""
Admin-only Telegram command bot.
Broadcast IDs receive signals but cannot issue commands.
"""
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from config import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_ID,
    SLST_START, SLST_END, SLST_UTC_OFFSET,
    SCAN_INTERVAL_MIN, TOP_COINS_LIMIT,
    OPENROUTER_MODEL, OPENROUTER_API_KEYS,
    OFF_WEEKDAYS,
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
    "enabled":      True,
}


def update_state(**kwargs):
    AGENT_STATE.update(kwargs)


def _is_admin(update: Update) -> bool:
    if not update.effective_chat:
        return False
    return str(update.effective_chat.id) == str(TELEGRAM_ADMIN_ID)


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


# ── commands ─────────────────────────────────────────────────────
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update):
        return
    await _reply(update,
        "🤖 *Scalp Hybrid Commands*\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "/on     — start scanning (no time limit)\n"
        "/off    — stop scanning (trade monitor keeps running)\n"
        "/status — agent health\n"
        "/trades — active + pending trades\n"
        "/pnl    — net performance (fees/slippage/funding)\n"
        "/stats  — same as /pnl\n"
        "/scan   — scan now (still obeys ON/OFF)\n"
        "/help   — this message"
    )


async def cmd_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update):
        return
    AGENT_STATE["enabled"] = True
    await _reply(update, "🟢 Scanning *ON* — a scan runs after every 5m candle close.")


async def cmd_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update):
        return
    AGENT_STATE["enabled"] = False
    await _reply(update, "🔴 Scanning *OFF* — open/pending trades are still tracked.")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update):
        return
    now = datetime.now(timezone.utc)
    started = AGENT_STATE.get("started_at")
    uptime = _fmt_dur((now - started).total_seconds()) if started else "?"
    ls = AGENT_STATE.get("last_scan_at")
    ls_str = ls.strftime("%H:%M:%S UTC") if ls else "—"
    sig = AGENT_STATE.get("last_signal")
    sig_str = (f"{sig['direction']} {sig['symbol']} ({sig.get('confidence','?')})" if sig else "—")
    on = AGENT_STATE.get("enabled", True)
    m = SCAN_INTERVAL_MIN - (now.minute % SCAN_INTERVAL_MIN)
    await _reply(update,
        f"🤖 *Scalp Hybrid Status*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{'🟢 *ON*' if on else '🔴 *OFF*'}\n"
        f"⏱ Uptime: `{uptime}`\n"
        f"📊 Scans: `{AGENT_STATE.get('scan_count', 0)}` · 🚀 Signals: `{AGENT_STATE.get('signal_count', 0)}`\n"
        f"🔍 Last scan: `{ls_str}`\n"
        f"📈 Last signal: `{sig_str}`\n"
        f"⏭ Next scan in ≈`{m}m`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🧠 `{OPENROUTER_MODEL}` · keys: `{len(OPENROUTER_API_KEYS)}`\n"
        f"🪙 `{TOP_COINS_LIMIT}` coins | `1H bias · 15m setup · 5m entry`"
    )


async def cmd_trades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update):
        return
    from trade_monitor import get_active_trades, get_pending_trades, get_prices, calc_pnl

    pending = get_pending_trades()
    active = get_active_trades()
    if not pending and not active:
        await _reply(update, "📭 No trades.")
        return

    symbols = list({t["symbol"] for t in pending + active})
    try:
        prices = await get_prices(symbols)
    except Exception:
        prices = {}

    lines = ["📊 *Trades*", "━━━━━━━━━━━━━━━━━━━━"]

    for t in pending:
        d = t["direction"].upper()
        emoji = "🟡"
        lines.append(
            f"{emoji} `{t['symbol']}` *{d}* LIMIT @ `{float(t['entry']):.6g}`\n"
            f"   Zone: `{float(t.get('zone_low') or 0):.6g}` – "
            f"`{float(t.get('zone_high') or 0):.6g}`"
        )

    for t in active:
        sym, d = t["symbol"], t["direction"].upper()
        entry = float(t["entry"])
        price = prices.get(sym)
        emoji = "🟢" if d == "LONG" else "🔴"
        pnl_str = (f"{calc_pnl(t, price)['pnl_pct']:+.2f}%"
                   if price else "N/A")
        status = t.get("status", "open")
        lines.append(
            f"{emoji} `{sym}` *{d}* `{status}` — PnL: `{pnl_str}`\n"
            f"   E: `{entry:.6g}` | SL: `{float(t['sl']):.6g}`\n"
            f"   TP1: `{float(t['tp1']):.6g}` | "
            f"TP2: `{float(t['tp2']):.6g}` | "
            f"TP3: `{float(t['tp3']):.6g}`"
        )

    await _reply(update, "\n".join(lines))


async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update):
        return
    from trade_monitor import get_closed_trades
    from performance import summarize

    s = summarize(get_closed_trades())
    if not s["signals_closed"]:
        await _reply(update, "📭 No closed trades.")
        return

    def f(x, fmt="{:.2f}"):
        return "n/a" if x is None else fmt.format(x)
    wr = "n/a" if s["win_rate"] is None else f"{s['win_rate'] * 100:.1f}% ({s['wins']}/{s['filled']})"
    fr = "n/a" if s["fill_rate"] is None else f"{s['fill_rate'] * 100:.0f}%"
    await _reply(update,
        f"📈 *Performance* _(simulated, net of costs)_\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Filled: `{s['filled']}` · expired `{s['unfilled_expired']}` · invalidated `{s['unfilled_invalidated']}`\n"
        f"Fill rate: `{fr}` · ambiguous: `{s['ambiguous']}`\n"
        f"Win rate: `{wr}`\n"
        f"Avg win/loss: `{f(s['avg_win_r'])}R` / `{f(s['avg_loss_r'])}R`\n"
        f"Expectancy: `{f(s['expectancy_r'], '{:+.2f}')}R` · PF: `{f(s['profit_factor'])}`\n"
        f"Total: `{s['total_r']:+.2f}R` · max DD: `{s['max_drawdown_r']:.2f}R`\n"
        f"Costs: fees `{s['fees_pct_sum']:.2f}%` slip `{s['slippage_pct_sum']:.2f}%` fund `{s['funding_pct_sum']:+.2f}%`\n"
        f"_Small samples are not evidence of an edge._"
    )


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update):
        return
    ev = AGENT_STATE.get("scan_trigger")
    if ev is None:
        await _reply(update, "⚠️ Manual scan not available.")
        return
    if not AGENT_STATE.get("enabled", True):
        await _reply(update, "🔴 Scanning is OFF — use /on first.")
        return
    ev.set()
    await _reply(update, "🚀 Manual scan queued (same validation and rate limits as scheduled scans).")


# ── lifecycle ────────────────────────────────────────────────────
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
    app.add_handler(CommandHandler("on",     cmd_on))
    app.add_handler(CommandHandler("off",    cmd_off))
    app.add_handler(CommandHandler("stats",  cmd_pnl))

    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)
    logger.info("Telegram bot started — admin-only commands")

    try:
        await asyncio.Event().wait()
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
