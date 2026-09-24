"""
DeepSeek agent — strict 5-confirmation SMC/ICT filter.

Thinking mode enabled with reasoning_effort="high" (default).
Retries on transient API errors.
"""
import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from openai import AsyncOpenAI

from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    REASONING_EFFORT, USE_THINKING,
)
from telegram_sender import send_signal

logger = logging.getLogger(__name__)

client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

_API_RETRIES = 3


SYSTEM_PROMPT = """You are an elite crypto futures day trader specialising in SMC/ICT,
liquidity, support/resistance, and premium/discount strategies.

You receive a compact JSON payload for the top 50 Binance Futures coins.
Each coin has 4H, 1H, and 15m sections.

## Task
Find the BEST trade setup across all coins. Emit AT MOST one signal per cycle.
If no valid setup exists, reply exactly "NO_SETUP".

## MANDATORY 5-Confirmation Rule
A signal is ONLY valid when ALL 5 confirmations align:
  1. Market structure (BOS/CHoCH) aligned across 2+ timeframes
  2. Price in discount (for LONG) or premium (for SHORT)
  3. Order Block OR FVG confluence nearby
  4. Liquidity sweep confirmation (recent)
  5. RSI supportive (not overbought for LONG / oversold for SHORT)

If ANY confirmation is missing, reply "NO_SETUP". Do NOT force trades.

## Confidence Tier
- All 5 aligned + 4H trend strongly directional -> "high"
- All 5 aligned + 4H trend moderate -> "medium"
- All 5 aligned but 15m entry imperfect -> "low"

## SL/TP Rules
- SL: just beyond nearest structure level (swing high/low, OB edge, bb edge)
  AND at least 0.5x ATR away from entry
- TP1: nearest liquidity level OR 1R (whichever is closer)
- TP2: 2R OR next S/R cluster
- TP3: 3R OR opposite-side liquidity

## Field Reference
bb/bbT/bbB   = Breaker Block
obL/fvgL     = top 3 recent Order Blocks / FVGs
obM/fvgM     = mitigation status
liqBS/liqSS  = buy-side / sell-side liquidity
sr           = top 5 S/R clusters
rsiDiv       = RSI divergence
rv           = relative volume
vah/val      = Value Area High/Low
oteL/oteS    = Optimal Trade Entry zones
sh2/sl2      = second-to-last swings

## Discipline
Quality over quantity. A cycle with no signal is perfectly acceptable.
Emit the reason field as a concise confluence summary (max 120 chars)."""


TOOLS = [{
    "type": "function",
    "function": {
        "name": "send_trade_signal",
        "description": "Send a confirmed trade signal to Telegram.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "e.g. BTCUSDT"},
                "direction": {"type": "string", "enum": ["LONG", "SHORT"]},
                "entry": {"type": "number"},
                "sl": {"type": "number"},
                "tp1": {"type": "number"},
                "tp2": {"type": "number"},
                "tp3": {"type": "number"},
                "confidence": {"type": "string",
                               "enum": ["low", "medium", "high"]},
                "reason": {"type": "string",
                           "description": "Concise confluence summary (max 120 chars)."},
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


def _save_reasoning(reasoning: str):
    try:
        os.makedirs("debug_reasoning", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        with open(f"debug_reasoning/{ts}.txt", "w", encoding="utf-8") as f:
            f.write(reasoning)
    except Exception as e:
        logger.warning("Could not save reasoning: %s", e)


async def run_agent(payload_json: str) -> Optional[Dict[str, Any]]:
    if not DEEPSEEK_API_KEY:
        logger.error("DEEPSEEK_API_KEY is not set")
        return None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Market data:\n{payload_json}"},
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
                logger.error("DeepSeek API failed after %d attempts: %s",
                             _API_RETRIES, exc)
                return None
            wait = 2 ** attempt
            logger.warning("DeepSeek error (attempt %d/%d): %s — retry in %ds",
                           attempt + 1, _API_RETRIES, exc, wait)
            await asyncio.sleep(wait)

    if response is None or not response.choices:
        logger.error("DeepSeek returned empty response")
        return None

    msg = response.choices[0].message

    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning:
        preview = reasoning[:500].replace("\n", " ")
        logger.info("🧠 Reasoning (%d chars): %s …", len(reasoning), preview)
        _save_reasoning(reasoning)

    if not msg.tool_calls:
        content = (msg.content or "").strip()
        if content:
            logger.info("Agent reply: %s", content[:200])
        else:
            logger.info("Agent decided NO_SETUP (reasoning=%d chars)",
                        len(reasoning) if reasoning else 0)
        return None

    tool_call = msg.tool_calls[0]
    args_str = tool_call.function.arguments or ""
    if not args_str:
        logger.error("Tool call had empty arguments")
        return None

    try:
        args = json.loads(args_str)
    except json.JSONDecodeError:
        logger.error("Failed to parse tool args: %s", args_str)
        return None

    logger.info("🚀 Signal: %s %s (confidence=%s)",
                args["direction"], args["symbol"], args.get("confidence"))

    if await send_signal(args):
        logger.info("Signal sent to Telegram for %s", args["symbol"])
        return args
    logger.error("Failed to send Telegram signal")
    return None
