"""
Telegram Bot API sender — broadcasts to all TELEGRAM_BROADCAST_IDS.
Returns True if at least one recipient received the message.
"""
import asyncio
import logging
from typing import List

import aiohttp

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_BROADCAST_IDS

logger = logging.getLogger(__name__)

_MD_SPECIALS = ("_", "*", "`", "[", "]")


def _escape_md(text: str) -> str:
    if not text:
        return ""
    for ch in _MD_SPECIALS:
        text = text.replace(ch, f"\\{ch}")
    return text


async def _send(text: str) -> bool:
    """Broadcast `text` to all TELEGRAM_BROADCAST_IDS.
    Returns True if at least one recipient succeeded."""
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not set")
        return False
    if not TELEGRAM_BROADCAST_IDS:
        logger.error("No TELEGRAM_BROADCAST_IDS configured")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload_base = {
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }

    successes = 0
    failures = 0

    async with aiohttp.ClientSession() as session:
        for chat_id in TELEGRAM_BROADCAST_IDS:
            try:
                async with session.post(
                    url, json={**payload_base, "chat_id": chat_id}, timeout=10
                ) as resp:
                    if resp.status == 200:
                        successes += 1
                    else:
                        failures += 1
                        body = await resp.text()
                        logger.error(
                            "Telegram send to %s failed (%d): %s",
                            chat_id, resp.status, body[:200],
                        )
            except Exception as exc:
                failures += 1
                logger.error("Telegram send to %s exception: %s", chat_id, exc)

    if failures:
        logger.warning("Broadcast: %d sent, %d failed (total=%d)",
                       successes, failures, len(TELEGRAM_BROADCAST_IDS))
    else:
        logger.info("Broadcast delivered to %d recipients", successes)

    return successes > 0


async def send_signal(signal: dict) -> bool:
    """Send a formatted trade signal to all recipients."""
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
    """Send a plain message (trade closed, status, etc.) to all recipients."""
    return await _send(text)
