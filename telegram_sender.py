"""
Sends signals + raw messages to Telegram via the Bot API.
Markdown escaping fixed to avoid Telegram 400 errors on special chars.
"""
import logging
import aiohttp
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)

# Characters that need escaping in legacy Markdown
_MD_SPECIALS = ("_", "*", "`", "[", "]")


def _escape_md(text: str) -> str:
    """Escape legacy Markdown special characters."""
    if not text:
        return ""
    for ch in _MD_SPECIALS:
        text = text.replace(ch, f"\\{ch}")
    return text


async def send_signal(signal: dict) -> bool:
    """Send a formatted trade signal to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Telegram credentials not configured")
        return False

    symbol     = signal.get("symbol", "???")
    direction  = signal.get("direction", "???").upper()
    entry      = signal.get("entry", "-")
    sl         = signal.get("sl", "-")
    tp1        = signal.get("tp1", "-")
    tp2        = signal.get("tp2", "-")
    tp3        = signal.get("tp3", "-")
    confidence = signal.get("confidence", "-")
    reason     = _escape_md(str(signal.get("reason", "")))

    emoji = "🟢" if direction == "LONG" else "🔴" if direction == "SHORT" else "⚪"

    text = (
        f"{emoji} *{direction}* | `{symbol}`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Entry: `{entry}`\n"
        f"🛑 SL:    `{sl}`\n"
        f"🎯 TP1:  `{tp1}`\n"
        f"🎯 TP2:  `{tp2}`\n"
        f"🎯 TP3:  `{tp3}`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 Confidence: `{confidence}`\n"
        f"📝 {reason}"
    )

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
        logger.error("Telegram send exception: %s", exc)
        return False


async def send_raw_message(text: str) -> bool:
    """Send a plain text message (status updates, trade closes)."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID,
               "text": text, "parse_mode": "Markdown"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=10) as resp:
                return resp.status == 200
    except Exception:
        return False
