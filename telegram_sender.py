"""
Broadcast sender — admin + friend IDs receive signals.
Only the admin ID can issue commands (enforced in telegram_bot.py).
"""
import asyncio
import logging
import aiohttp

from config import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_ID, TELEGRAM_BROADCAST_IDS,
)

logger = logging.getLogger(__name__)

_MD_SPECIALS = ("_", "*", "`", "[", "]")


def _escape_md(text: str) -> str:
    if not text:
        return ""
    for ch in _MD_SPECIALS:
        text = text.replace(ch, f"\\{ch}")
    return text


def _broadcast_ids() -> list[str]:
    ids = [TELEGRAM_ADMIN_ID] + TELEGRAM_BROADCAST_IDS
    seen, out = set(), []
    for x in ids:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


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
            async with session.post(url, json=payload, timeout=10) as r:
                if r.status != 200:
                    body = await r.text()
                    logger.error("Telegram to %s failed %s: %s",
                                 chat_id, r.status, body[:200])
                    return False
                return True
    except Exception as exc:
        logger.error("Telegram exception to %s: %s", chat_id, exc)
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
    reason = _escape_md(str(signal.get("reason", "")))

    zone_line = ""
    if entry_type == "LIMIT" and signal.get("zone_high") and signal.get("zone_low"):
        zone_line = (f"📦 Zone: `{signal['zone_low']}` – "
                     f"`{signal['zone_high']}`\n")

    text = (
        f"{emoji} *{direction}* | `{sym}`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⚡ Type: `{entry_type}`\n"
        f"💰 Entry: `{signal.get('entry', '-')}`\n"
        f"{zone_line}"
        f"🛑 SL:    `{signal.get('sl', '-')}`\n"
        f"🎯 TP1:  `{signal.get('tp1', '-')}`\n"
        f"🎯 TP2:  `{signal.get('tp2', '-')}`\n"
        f"🎯 TP3:  `{signal.get('tp3', '-')}`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 Confidence: `{signal.get('confidence', '-')}`\n"
        f"📈 R:R: `{signal.get('rr_ratio', '-')}`\n"
        f"📝 {reason}"
    )
    return await _broadcast(text)


async def send_raw_message(text: str) -> bool:
    return await _broadcast(text)
