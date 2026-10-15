"""
DeepSeek Flash vision agent — hybrid text + charts.
With payload size validation and async file encoding.
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
    REASONING_EFFORT, USE_THINKING, CHART_MAX_TOTAL_MB,
)
from telegram_sender import send_signal

logger = logging.getLogger(__name__)
client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
_API_RETRIES = 3


SYSTEM_PROMPT = """You are an ICT/SMC liquidity trader for Binance Futures day trading.

## STRATEGY (only these)
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

## SL / TP — use EXACT prices
- `raw` field in each TF section = last 20 candles [ts,o,h,l,c,v]
- Use these raw candles for precise SL/TP prices
- SL: just beyond the nearest liquidity pool (swept low/high, OB edge,
  swing point, bb edge) that invalidates the setup
- TP1: nearest opposing liquidity level
- TP2: next S/R cluster or 2R
- TP3: 3R or opposite-side liquidity

## RULES
- Both LONG and SHORT allowed
- Charts (4H, 1H, 15m) show visual context — study sweeps, wicks,
  displacement, and structure at the right edge
- If no valid setup: reply exactly NO_SETUP. Do not force trades.

## OUTPUT
Use send_trade_signal tool. `reason` = concise confluence summary (max 120 chars)."""


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
                "confidence": {"type": "string",
                               "enum": ["low", "medium", "high"]},
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


def _encode_chart_sync(path: str) -> Optional[str]:
    """Sync base64 encoder — called via asyncio.to_thread."""
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


async def _encode_all_charts(
    chart_map: Optional[Dict[str, List[str]]],
) -> List[str]:
    """Encode all charts concurrently, with size logging."""
    if not chart_map:
        return []

    tasks = []
    for symbol, paths in chart_map.items():
        for path in paths:
            tasks.append(asyncio.to_thread(_encode_chart_sync, path))

    if not tasks:
        return []

    results = await asyncio.gather(*tasks, return_exceptions=True)
    urls = [r for r in results if isinstance(r, str)]

    total_bytes = sum(len(u) for u in urls)
    total_mb = total_bytes / (1024 * 1024)
    logger.info("Encoded %d charts, total %.1f MB base64",
                len(urls), total_mb)

    if total_mb > CHART_MAX_TOTAL_MB:
        logger.warning("Payload %.1f MB exceeds safe limit (%d MB) — "
                       "consider reducing CHART_COIN_LIMIT",
                       total_mb, CHART_MAX_TOTAL_MB)
    return urls


async def run_agent(
    payload_json: str,
    chart_map: Optional[Dict[str, List[str]]] = None,
) -> Optional[Dict[str, Any]]:
    if not DEEPSEEK_API_KEY:
        logger.error("DEEPSEEK_API_KEY missing")
        return None

    # Encode charts asynchronously
    chart_urls = await _encode_all_charts(chart_map)

    content: List[Dict[str, Any]] = [
        {"type": "text", "text": f"Market data:\n{payload_json}"},
    ]
    for url in chart_urls:
        content.append({
            "type": "image_url",
            "image_url": {"url": url},
        })

    if not chart_urls:
        logger.warning("No charts encoded — sending text-only payload")
    else:
        logger.info("Sending %d charts + text to DeepSeek …", len(chart_urls))

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

        # ── Log token usage ─────────────────────────────────────────
    try:
        usage = response.usage
        if usage:
            logger.info("📊 Tokens — prompt: %d, completion: %d, total: %d",
                        usage.prompt_tokens or 0,
                        usage.completion_tokens or 0,
                        usage.total_tokens or 0)
    except Exception:
        pass

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
