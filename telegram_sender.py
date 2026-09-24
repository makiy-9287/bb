"""
Telegram Bot API sender — fixed Markdown escaping.
"""
import logging
import aiohttp

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)

_MD_SPECIALS = ("_", "*", "`", "[", "]")


def _escape_md(text: str) -> str:
    if not text:
        return ""
    for ch in _MD_SPECIALS:
        text = text.replace(ch, f"\\{ch}")
    return text


async def _send(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Telegram credentials not configured")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=10) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error("Telegram send failed %s: %s",
                                 resp.status, body)
                    return False
                return True
    except Exception as exc:
        logger.error("Telegram exception: %s", exc)
        return False


async def send_signal(signal: dict) -> bool:
    symbol     = signal.get("symbol", "???")
    direction  = signal.get("direction", "???").upper()
    emoji = "🟢" if direction == "LONG" else "🔴" if direction == "SHORT" else "⚪"
    reason = _escape_md(str(signal.get("reason", "")))

    text = (
        f"{emoji} *{direction}* | `{symbol}`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Entry: `{signal.get('entry', '-')}`\n"
        f"🛑 SL:    `{signal.get('sl', '-')}`\n"
        f"🎯 TP1:  `{signal.get('tp1', '-')}`\n"
        f"🎯 TP2:  `{signal.get('tp2', '-')}`\n"
        f"🎯 TP3:  `{signal.get('tp3', '-')}`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 Confidence: `{signal.get('confidence', '-')}`\n"
        f"📝 {reason}"
    )
    return await _send(text)


async def send_raw_message(text: str) -> bool:
    return await _send(text)
