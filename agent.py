"""
The DeepSeek agent.  Receives the compact JSON payload, calls
deepseek-flash with thinking mode + reasoning_effort=medium, and
executes any trade signals the model emits.
"""
import json
import logging
from typing import Any, Dict, Optional

from openai import AsyncOpenAI

from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    REASONING_EFFORT, USE_THINKING
)
from telegram_sender import send_signal

logger = logging.getLogger(__name__)

client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

SYSTEM_PROMPT = """You are an elite crypto futures trader specialising in SMC/ICT,
liquidity, support/resistance, and premium/discount strategies.

You will receive a compact JSON payload containing multi-timeframe data for the
top 70 Binance Futures coins. Each coin has 1D, 4H, and 1H sections.

Think through the setups step by step before deciding. Analyse:
1. Market structure alignment (BOS/CHoCH) across timeframes
2. Premium/discount positioning
3. Order Block + FVG confluence
4. Liquidity sweeps / pools
5. RSI momentum confirmation

A valid setup MUST have CONFLUENCE across timeframes:
   - Market structure (BOS/CHoCH) aligned
   - Price in discount (for longs) or premium (for shorts)
   - Order Block or FVG confluence
   - Liquidity sweep confirmation
   - RSI not overbought/oversold against direction

If a setup exists, call the `send_trade_signal` tool with exact values.
If NO valid setup exists, simply reply "NO_SETUP". Do not force a trade.

SL/TP rules:
- SL must be placed beyond the nearest structure level (swing high/low or OB edge).
- TP1 = 1R, TP2 = 2R, TP3 = 3R from entry, adjusted to nearest liquidity level.
- Use ATR to ensure SL is not too tight (minimum 0.5× ATR away).

Be extremely selective. Only the highest-probability setup should be emitted."""

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
                        "description": "Concise SMC/ICT confluence summary (max 120 chars).",
                    },
                },
                "required": ["symbol", "direction", "entry", "sl",
                             "tp1", "tp2", "tp3", "confidence", "reason"],
            },
        },
    }
]


def _build_request_kwargs() -> dict:
    """Assemble the kwargs for the DeepSeek call, toggling thinking mode."""
    kwargs: Dict[str, Any] = {
        "model": DEEPSEEK_MODEL,
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": 1200,
    }

    if USE_THINKING:
        # Thinking mode: temperature/top_p/penalties are ignored by DeepSeek
        # per official docs. We therefore do NOT pass temperature at all.
        kwargs["reasoning_effort"] = REASONING_EFFORT
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        logger.info("Thinking mode ON, reasoning_effort=%s", REASONING_EFFORT)
    else:
        # Non-thinking fallback: temperature is respected here
        kwargs["temperature"] = 0.2

    return kwargs


async def run_agent(payload_json: str) -> Optional[Dict[str, Any]]:
    """
    Send the payload to DeepSeek.  If the model returns a tool call,
    execute it (send Telegram signal) and return the signal dict.
    Otherwise return None.
    """
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

    # ── Log the chain-of-thought (if present) ───────────────────
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning:
        preview = reasoning[:500].replace("\n", " ")
        logger.info("🧠 Reasoning (truncated): %s …", preview)

        # ── Model chose NOT to trade ────────────────────────────────
    if not msg.tool_calls:
        content = (msg.content or "").strip()
        if content:
            logger.info("Agent reply: %s", content[:200])
        else:
            # Thinking mode often leaves `content` empty when the
            # model decides there's nothing to trade.
            logger.info("Agent decided NO_SETUP (empty content, "
                        "reasoning length=%d chars)",
                        len(reasoning) if reasoning else 0)
        return None

    # ── Model emitted a tool call ───────────────────────────────
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

    logger.info("Agent signal: %s %s", args["direction"], args["symbol"])

    sent = await send_signal(args)
    if sent:
        logger.info("Signal sent to Telegram for %s", args["symbol"])
        return args
    else:
        logger.error("Failed to send Telegram signal")
        return None
