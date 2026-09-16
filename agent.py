"""
DeepSeek agent — strict 5-confirmation SMC/ICT entry filter.

Reasoning effort = high (per DeepSeek docs, thinking mode default is
high; medium maps to high, so "high" is the effective ceiling for
non-max requests).
"""
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from openai import AsyncOpenAI

from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    REASONING_EFFORT, USE_THINKING
)
from telegram_sender import send_signal

logger = logging.getLogger(__name__)

client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


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
- All 5 aligned + HTF (4H) trend strongly directional -> "high"
- All 5 aligned + HTF trend moderate -> "medium"
- All 5 aligned but 15m entry imperfect -> "low"

## SL/TP Rules
- SL: just beyond nearest structure level (swing high/low, OB edge, or bb edge)
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


TOOLS = [
    {
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
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                    "reason": {
                        "type": "string",
                        "description": "Concise confluence summary (max 120 chars).",
                    },
                },
                "required": ["symbol", "direction", "entry", "sl",
                             "tp1", "tp2", "tp3", "confidence", "reason"],
            },
        },
    }
]


def _build_request_kwargs() -> dict:
    """Assemble kwargs for DeepSeek. Thinking mode ignores temperature."""
    kwargs: Dict[str, Any] = {
        "model": DEEPSEEK_MODEL,
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": 2000,
    }

    if USE_THINKING:
        # Per DeepSeek docs, thinking mode is enabled by default with
        # default effort = high. We set it explicitly.
        kwargs["reasoning_effort"] = REASONING_EFFORT
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        logger.info("Thinking mode ON, reasoning_effort=%s", REASONING_EFFORT)
    else:
        kwargs["temperature"] = 0.2

    return kwargs


def _save_reasoning(reasoning: str):
    """Save full reasoning trace to disk for inspection."""
    try:
        debug_dir = "debug_reasoning"
        os.makedirs(debug_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = f"{debug_dir}/{ts}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(reasoning)
        logger.debug("Reasoning saved → %s", path)
    except Exception as e:
        logger.warning("Could not save reasoning: %s", e)


async def run_agent(payload_json: str) -> Optional[Dict[str, Any]]:
    """Send payload to DeepSeek and execute any emitted signal."""
    if not DEEPSEEK_API_KEY:
        logger.error("DEEPSEEK_API_KEY is not set")
        return None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Market data:\n{payload_json}"},
    ]

    try:
        response = await client.chat.completions.create(
            messages=messages,
            **_build_request_kwargs(),
        )
    except Exception as exc:
        logger.error("DeepSeek API error: %s", exc)
        return None

    choice = response.choices[0]
    msg = choice.message

    # ── Log + save reasoning ────────────────────────────────────
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning:
        preview = reasoning[:500].replace("\n", " ")
        logger.info("🧠 Reasoning (%d chars, truncated): %s …",
                    len(reasoning), preview)
        _save_reasoning(reasoning)

    # ── No trade ────────────────────────────────────────────────
    if not msg.tool_calls:
        content = (msg.content or "").strip()
        if content:
            logger.info("Agent reply: %s", content[:200])
        else:
            logger.info("Agent decided NO_SETUP (reasoning=%d chars)",
                        len(reasoning) if reasoning else 0)
        return None

    # ── Trade signal ────────────────────────────────────────────
    tool_call = msg.tool_calls[0]
    args_str = tool_call.function.arguments or ""
    if not args_str:
        logger.error("Tool call had empty arguments")
        return None

    try:
        args = json.loads(args_str)
    except json.JSONDecodeError:
        logger.error("Failed to parse tool arguments: %s", args_str)
        return None

    logger.info("🚀 Agent signal: %s %s (confidence=%s)",
                args["direction"], args["symbol"], args.get("confidence"))

    sent = await send_signal(args)
    if sent:
        logger.info("Signal sent to Telegram for %s", args["symbol"])
        return args
    logger.error("Failed to send Telegram signal")
    return None
