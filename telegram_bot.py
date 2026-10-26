"""
Admin-only Telegram command bot (V10). Nobody except TELEGRAM_ADMIN_ID can issue commands or receive messages.
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from config import (TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_ID, SCAN_INTERVAL_MIN, TOP_COINS_LIMIT, OPENROUTER_MODEL,
                    OPENROUTER_API_KEYS, COINS_PER_CALL)
from schedule import in_window, next_window_start, now_local

logger = logging.getLogger(__name__)

AGENT_STATE: Dict[str, Any] = {
    "started_at": None, "last_scan_at": None, "scan_count": 0, "signal_count": 0,
    "last_signal": None, "scan_trigger": None, "enabled": True,
}


def update_state(**kwargs):
    AGENT_STATE.update(kwargs)


def _is_admin(update: Update) -> bool:
    return bool(update.effective_chat) and str(update.effective_chat.id) == str(TELEGRAM_ADMIN_ID)


def _fmt_dur(seconds: float) -> str:
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"
    return f"{int(seconds // 60)}m"


async def _reply(update: Update, text: str) -> None:
    try:
        if update.message:
            await update.message.reply_text(text, parse_mode="Markdown")
        elif update.effective_chat:
            await update.get_bot().send_message(chat_id=update.effective_chat.id, text=text, parse_mode="Markdown")
    except Exception as exc:
        logger.warning("Reply failed: %s", str(exc).replace(TELEGRAM_BOT_TOKEN, "<token>") if TELEGRAM_BOT_TOKEN else exc)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update):
        return
    await _reply(update,
        "🤖 *V10 S/R Agent*\n━━━━━━━━━━━━━━━━━━━━\n"
        "/on     — scanning ON (inside the window)\n"
        "/off    — scanning OFF (trade tracking continues)\n"
        "/status — health + window\n"
        "/trades — active + pending trades\n"
        "/pnl    — net performance (simulated)\n"
        "/scan   — scan now (inside the window only)\n"
        "/help   — this message")


async def cmd_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update):
        return
    AGENT_STATE["enabled"] = True
    await _reply(update, "🟢 Scanning *ON* (runs every 15 min inside 12:00–19:00 Colombo).")


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
    sig_str = f"{sig['direction']} {sig['symbol']} (q{sig.get('quality', '?')})" if sig else "—"
    on, win = AGENT_STATE.get("enabled", True), in_window()
    nxt = "open" if win else "closed, next " + next_window_start().strftime("%a %H:%M")
    await _reply(update,
        f"🤖 *V10 S/R Agent*\n━━━━━━━━━━━━━━━━━━━━\n"
        f"{'🟢 *ON*' if on else '🔴 *OFF*'} · window 12:00–19:00 Colombo: {nxt}\n"
        f"⏱ Uptime: `{uptime}` · Colombo `{now_local().strftime('%a %H:%M')}`\n"
        f"📊 Scans: `{AGENT_STATE.get('scan_count', 0)}` · 🚀 Signals: `{AGENT_STATE.get('signal_count', 0)}`\n"
        f"🔍 Last scan: `{ls_str}`\n📈 Last signal: `{sig_str}`\n━━━━━━━━━━━━━━━━━━━━\n"
        f"🧠 `{OPENROUTER_MODEL}` · keys `{len(OPENROUTER_API_KEYS)}`\n"
        f"🪙 `{TOP_COINS_LIMIT}` coins · `{COINS_PER_CALL}`/call · every `{SCAN_INTERVAL_MIN}m` · `4h/1h/15m`")


async def cmd_trades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update):
        return
    from trade_monitor import get_active_trades, get_pending_trades, get_prices, calc_pnl
    pending, active = get_pending_trades(), get_active_trades()
    if not pending and not active:
        await _reply(update, "📭 No trades.")
        return
    try:
        prices = await get_prices(list({t["symbol"] for t in pending + active}))
    except Exception:
        prices = {}
    lines = ["📊 *Trades*", "━━━━━━━━━━━━━━━━━━━━"]
    for t in pending:
        lines.append(f"🟡 `{t['symbol']}` *{t['direction'].upper()}* LIMIT @ `{float(t['entry']):.6g}`\n"
                     f"   Zone: `{float(t.get('zone_low') or 0):.6g}` – `{float(t.get('zone_high') or 0):.6g}`")
    for t in active:
        sym, d = t["symbol"], t["direction"].upper()
        px = prices.get(sym)
        pnl = f"{calc_pnl(t, px)['pnl_pct']:+.2f}%" if px else "N/A"
        lines.append(f"{'🟢' if d == 'LONG' else '🔴'} `{sym}` *{d}* `{t.get('status', 'open')}` — PnL: `{pnl}`\n"
                     f"   E: `{float(t['entry']):.6g}` | SL: `{float(t['sl']):.6g}`\n"
                     f"   TP1: `{float(t['tp1']):.6g}` | TP2: `{float(t['tp2']):.6g}` | TP3: `{float(t['tp3']):.6g}`")
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
    f = lambda x, fmt="{:.2f}": "n/a" if x is None else fmt.format(x)
    wr = "n/a" if s["win_rate"] is None else f"{s['win_rate'] * 100:.1f}% ({s['wins']}/{s['filled']})"
    fr = "n/a" if s["fill_rate"] is None else f"{s['fill_rate'] * 100:.0f}%"
    await _reply(update,
        f"📈 *Performance* _(simulated, net of costs)_\n━━━━━━━━━━━━━━━━━━━━\n"
        f"Filled: `{s['filled']}` · expired `{s['unfilled_expired']}` · invalidated `{s['unfilled_invalidated']}`\n"
        f"Fill rate: `{fr}` · ambiguous: `{s['ambiguous']}`\nWin rate: `{wr}`\n"
        f"Avg win/loss: `{f(s['avg_win_r'])}R` / `{f(s['avg_loss_r'])}R`\n"
        f"Expectancy: `{f(s['expectancy_r'], '{:+.2f}')}R` · PF: `{f(s['profit_factor'])}`\n"
        f"Total: `{s['total_r']:+.2f}R` · max DD: `{s['max_drawdown_r']:.2f}R`\n"
        f"Costs: fees `{s['fees_pct_sum']:.2f}%` slip `{s['slippage_pct_sum']:.2f}%` fund `{s['funding_pct_sum']:+.2f}%`\n"
        f"_Small samples are not evidence of an edge._")


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
    if not in_window():
        await _reply(update, f"⚪ Outside the scan window (12:00–19:00 Colombo). Next: {next_window_start().strftime('%a %H:%M')}")
        return
    ev.set()
    await _reply(update, "🚀 Manual scan queued.")


async def telegram_bot_loop():
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN not set — bot disabled")
        return
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    for name, fn in (("help", cmd_help), ("start", cmd_help), ("status", cmd_status), ("trades", cmd_trades),
                     ("pnl", cmd_pnl), ("stats", cmd_pnl), ("scan", cmd_scan), ("on", cmd_on), ("off", cmd_off)):
        app.add_handler(CommandHandler(name, fn))
    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)
    logger.info("Telegram bot started — admin-only")
    try:
        await asyncio.Event().wait()
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
