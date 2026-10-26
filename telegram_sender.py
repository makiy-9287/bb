"""
Telegram sender — ADMIN CHAT ONLY (signals and trade updates). Nobody else ever receives anything.
"""
import asyncio
import logging
import aiohttp

from config import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_ID, SIGNAL_EXPIRY_MIN,
)

logger = logging.getLogger(__name__)

_MD_SPECIALS = ("_", "*", "`", "[", "]")


def _redact(msg: str) -> str:
    """Never let the bot token reach the logs (aiohttp errors embed the request URL)."""
    return msg.replace(TELEGRAM_BOT_TOKEN, "<token>") if TELEGRAM_BOT_TOKEN else msg


def _escape_md(text: str) -> str:
    if not text:
        return ""
    for ch in _MD_SPECIALS:
        text = text.replace(ch, f"\\{ch}")
    return text


def _broadcast_ids() -> list[str]:
    """Admin only. (Kept under this name for backward compatibility.)"""
    return [TELEGRAM_ADMIN_ID] if TELEGRAM_ADMIN_ID else []


async def _send_to(chat_id: str, text: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload,
                                    timeout=aiohttp.ClientTimeout(total=10)) as r:
                if r.status == 200:
                    return True
                body = await r.text()
                logger.error("Telegram to %s failed %s: %s",
                             chat_id, r.status, _redact(body[:200]))
                if r.status == 400 and "parse" in body.lower():
                    # Markdown entity error → resend as plain text so signal isn't lost
                    plain = dict(payload)
                    plain.pop("parse_mode", None)
                    plain["text"] = text.replace("`", "").replace("*", "").replace("\\", "")
                    async with session.post(url, json=plain,
                                            timeout=aiohttp.ClientTimeout(total=10)) as r2:
                        return r2.status == 200
                return False
    except Exception as exc:
        logger.error("Telegram exception to %s: %s", chat_id, _redact(str(exc)))
        return False


async def _broadcast(text: str) -> bool:
    ids = _broadcast_ids()
    if not ids:
        logger.error("No Telegram chat IDs configured")
        return False
    results = await asyncio.gather(
        *[_send_to(cid, text) for cid in ids],
        return_exceptions=True,
    )
    return any(r is True for r in results)


async def send_signal(signal: dict) -> bool:
    sym = signal.get("symbol", "???")
    direction = signal.get("direction", "???").upper()
    entry_type = signal.get("entry_type", "CMP")
    emoji = "🟢" if direction == "LONG" else "🔴" if direction == "SHORT" else "⚪"
    zone_line = ""
    if signal.get("zone_high") and signal.get("zone_low"):
        zone_line = f"📦 Zone: `{signal['zone_low']}` – `{signal['zone_high']}`\n"
    exp = f"⏳ Limit expires in `{SIGNAL_EXPIRY_MIN}m` if unfilled\n" if entry_type == "LIMIT" else ""
    text = (
        f"{emoji} *{direction}* | `{sym}`  _(S/R zone)_\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⚡ Type: `{entry_type}`\n"
        f"💰 Entry: `{signal.get('entry', '-')}`\n"
        f"{zone_line}"
        f"🛑 SL:    `{signal.get('sl', '-')}`\n"
        f"🎯 TP1:  `{signal.get('tp1', '-')}`\n"
        f"🎯 TP2:  `{signal.get('tp2', '-')}`\n"
        f"🎯 TP3:  `{signal.get('tp3', '-')}`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 Net R:R (TP1, after costs): `{signal.get('net_rr', '-')}`\n"
        f"⭐ Quality (model's own 0-100, not a win probability): `{signal.get('quality', '-')}`\n"
        f"{exp}"
        f"🧭 {_escape_md(str(signal.get('confluence', '')))}\n"
        f"✅ {_escape_md(str(signal.get('confirmation', '')))}\n"
        f"📝 {_escape_md(str(signal.get('reason', '')))}"
    )
    return await _broadcast(text)


async def send_raw_message(text: str) -> bool:
    return await _broadcast(text)
