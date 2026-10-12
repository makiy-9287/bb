"""
Telegram broadcast sender — all recipients in TELEGRAM_BROADCAST_IDS.
"""
import logging
from typing import Any

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
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN missing")
        return False
    if not TELEGRAM_BROADCAST_IDS:
        logger.error("No broadcast recipients")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    base = {"text": text, "parse_mode": "Markdown",
            "disable_web_page_preview": True}

    ok = fail = 0
    async with aiohttp.ClientSession() as session:
        for chat_id in TELEGRAM_BROADCAST_IDS:
            try:
                async with session.post(
                    url, json={**base, "chat_id": chat_id}, timeout=10
                ) as resp:
                    if resp.status == 200:
                        ok += 1
                    else:
                        fail += 1
                        body = await resp.text()
                        logger.error("Telegram %s failed (%d): %s",
                                     chat_id, resp.status, body[:200])
            except Exception as exc:
                fail += 1
                logger.error("Telegram %s exception: %s", chat_id, exc)

    if fail:
        logger.warning("Broadcast: %d ok, %d fail (of %d)",
                       ok, fail, len(TELEGRAM_BROADCAST_IDS))
    else:
        logger.info("Broadcast delivered to %d", ok)
    return ok > 0


async def send_signal(signal: dict) -> bool:
    sym  = signal.get("symbol", "???")
    d    = signal.get("direction", "???").upper()
    emoji = "🟢" if d == "LONG" else "🔴" if d == "SHORT" else "⚪"
    reason = _escape_md(str(signal.get("reason", "")))

    text = (
        f"{emoji} *{d}* | `{sym}`\n"
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
