"""
DeepSeek Flash vision agent — hybrid text + charts.
Model: deepseek-flash (native vision, 384 tokens/image, max 600 images).
"""
import asyncio
import base64
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    REASONING_EFFORT, USE_THINKING,
)
from telegram_sender import send_signal

logger = logging.getLogger(__name__)

client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
_API_RETRIES = 3


SYSTEM_PROMPT = """You are an ICT/SMC liquidity trader for Binance Futures day trading.

## STRATEGY (only these — ignore everything else)
- Order Blocks (OB), Fair Value Gaps (FVG), Breaker Blocks (BB)
- Break of Structure (BOS), Change of Character (CHoCH)
- Liquidity sweeps, Equal Highs/Lows, Session extremes
- Premium/Discount zones, OTE (0.618–0.79)

## ENTRY — ALL 5 must align
1. BOS/CHoCH aligned on 2+ timeframes
2. Price in discount (LONG) or premium (SHORT)
3. OB or FVG or BB confluence nearby
4. Recent liquidity sweep (swB or swS with bar ≤ 8)
5. RSI supportive (not overbought for LONG / oversold for SHORT)

## SL / TP — use EXACT prices from the raw candles
- SL: just beyond the nearest liquidity pool that invalidates the setup
  (swept low/high, OB edge, swing point, bb edge)
- TP1: nearest opposing liquidity level
- TP2: next S/R cluster or 2R
- TP3: 3R or opposite-side liquidity

## RULES
- Both LONG and SHORT allowed — pick whichever has the confluence
- Charts (4H, 1H, 15m) give you the visual context: study sweeps, wicks,
  displacement, and structure at the right edge of each chart
- The `raw` field in the payload gives 20 exact candles per TF — use
  these for precise SL/TP prices
- If no valid setup exists: reply exactly NO_SETUP. Do not force trades.

## OUTPUT
Use the send_trade_signal tool. The `reason` field must be a concise
confluence summary (max 120 chars)."""


TOOLS = [{
    "type": "function",
    "function": {
        "name": "send_trade_signal",
        "description": "Send a confirmed trade signal to Telegram.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol":     {"type": "string"},
                "direction":  {"type": "string", "enum": ["LONG", "SHORT"]},
                "entry":      {"type": "number"},
                "sl":         {"type": "number"},
                "tp1":        {"type": "number"},
                "tp2":        {"type": "number"},
                "tp3":        {"type": "number"},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                "reason":     {"type": "string"},
            },
            "required": ["symbol", "direction", "entry", "sl",
                         "tp1", "tp2", "tp3", "confidence", "reason"],
        },
    },
}]


def _build_request_kwargs() -> dict:
    kwargs: Dict[str, Any] = {
        "model": DEEPSEEK_MODEL,
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": 16000,
    }
    if USE_THINKING:
        kwargs["reasoning_effort"] = REASONING_EFFORT
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        logger.info("Thinking ON, reasoning_effort=%s", REASONING_EFFORT)
    else:
        kwargs["temperature"] = 0.2
    return kwargs


def _encode_chart(path: str) -> Optional[str]:
    """Base64-encode a PNG chart as a data URL."""
    try:
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception as exc:
        logger.warning("Chart encode failed %s: %s", path, exc)
        return None


def _save_reasoning(reasoning: str):
    try:
        os.makedirs("debug_reasoning", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        with open(f"debug_reasoning/{ts}.txt", "w", encoding="utf-8") as f:
            f.write(reasoning)
    except Exception as e:
        logger.warning("Could not save reasoning: %s", e)


async def run_agent(
    payload_json: str,
    chart_map: Dict[str, List[str]],
) -> Optional[Dict[str, Any]]:
    """
    Send payload + charts to DeepSeek. chart_map = {symbol: [paths]}.
    """
    if not DEEPSEEK_API_KEY:
        logger.error("DEEPSEEK_API_KEY missing")
        return None

    # Build multimodal content blocks
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": f"Market data:\n{payload_json}"},
    ]

    chart_count = 0
    for symbol, paths in chart_map.items():
        for path in paths:
            url = _encode_chart(path)
            if url:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": url},
                })
                chart_count += 1

    logger.info("Sending %d charts + text to DeepSeek …", chart_count)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    response = None
    for attempt in range(_API_RETRIES):
        try:
            response = await client.chat.completions.create(
                messages=messages, **_build_request_kwargs(),
            )
            break
        except Exception as exc:
            if attempt == _API_RETRIES - 1:
                logger.error("DeepSeek failed after %d attempts: %s",
                             _API_RETRIES, exc)
                return None
            wait = 2 ** attempt
            logger.warning("DeepSeek err (attempt %d/%d): %s — retry %ds",
                           attempt + 1, _API_RETRIES, exc, wait)
            await asyncio.sleep(wait)

    if response is None or not response.choices:
        logger.error("DeepSeek returned empty")
        return None

    msg = response.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning:
        preview = reasoning[:500].replace("\n", " ")
        logger.info("🧠 Reasoning (%d chars): %s …", len(reasoning), preview)
        _save_reasoning(reasoning)

    if not msg.tool_calls:
        content_str = (msg.content or "").strip()
        if content_str:
            logger.info("Agent reply: %s", content_str[:200])
        else:
            logger.info("Agent NO_SETUP (reasoning=%d chars)",
                        len(reasoning) if reasoning else 0)
        return None

    args_str = msg.tool_calls[0].function.arguments or ""
    if not args_str:
        logger.error("Empty tool args")
        return None

    try:
        args = json.loads(args_str)
    except json.JSONDecodeError:
        logger.error("Bad tool args: %s", args_str)
        return None

    logger.info("🚀 Signal: %s %s (%s)",
                args["direction"], args["symbol"], args.get("confidence"))

    if await send_signal(args):
        logger.info("Signal broadcast for %s", args["symbol"])
        return args
    logger.error("Telegram send failed")
    return None
