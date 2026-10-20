"""
v4.4 Agent — 10-Point Scoring System for A+ Setups.
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


SYSTEM_PROMPT = """
You are an elite ICT/SMC trader managing a hedge fund.
Your ONLY job is to find the absolute BEST setups. Capital preservation is priority #1.
You analyze a batch of coins and score them. You ONLY send a signal if the score is 8 or higher (out of 10).

## DATA FORMAT
You get up to 5 coins per batch. Each coin has:
- 4h (150), 1h (100), 15m (100) candles: [high, low, close, volume].

## THE 10-POINT SCORING SYSTEM
A setup MUST score 8, 9, or 10 to be valid. If it scores 7 or below, you MUST reply with "NO_SETUP".

1. **Market Structure (2 pts):**
   - 2 pts: Clear BOS/CHoCH on BOTH 15m and 1H in the same direction.
   - 1 pt: Clear BOS/CHoCH on 15m only, or conflicting 1H structure.
   - 0 pts: Choppy or against 1H structure.

2. **Liquidity Sweep (2 pts):**
   - 2 pts: Recent sweep of a MAJOR liquidity level (e.g., 1H swing high/low, Equal Highs/Lows) followed by a strong rejection (wick) and 15m CHoCH.
   - 1 pt: Sweep of a minor 15m level.
   - 0 pts: No sweep, or price just drifting into a zone.

3. **Order Block / FVG Confluence (2 pts):**
   - 2 pts: Price is tapping into a fresh, high-volume 1H or 15m Order Block (OB) or Fair Value Gap (FVG) with immediate rejection.
   - 1 pt: Price is near an OB/FVG but hasn't shown a clear 15m rejection yet.
   - 0 pts: No OB/FVG nearby.

4. **Premium / Discount & POC (2 pts):**
   - 2 pts: Price is DEEP in Discount for LONG (below 50% of 4H range) or DEEP in Premium for SHORT. AND price is near the POC or Value Area (VAH/VAL).
   - 1 pt: Price is in Discount/Premium, but not near POC.
   - 0 pts: Price is in Equilibrium or wrong zone.

5. **Momentum & Indicators (2 pts):**
   - 2 pts: RSI is supportive (e.g., <40 for LONG, >60 for SHORT, or clear divergence). MA(7) is crossing MA(25) in trade direction. Volume spike confirms the move.
   - 1 pt: Mixed indicators (e.g., RSI ok, but no volume).
   - 0 pts: RSI is overbought for LONG or oversold for SHORT, or MAs are against you.

## HOW TO CALCULATE SL / TP (ONLY IF SCORE >= 8)
- **SL:** Place 3-5 ticks beyond the structural invalidation point (the swept low/high or OB edge) + 0.2x ATR buffer.
- **TP1:** Nearest 1H opposing liquidity (1R-1.5R).
- **TP2:** Next 1H swing extreme (2R).
- **TP3:** 3R.

## RULES
- Only one signal per batch.
- ALL signals MUST be labeled as "high" confidence.
- If you scan all 5 coins and the best one only scores 7/10, reply exactly "NO_SETUP".
- Do not force trades.

## OUTPUT
Use the send_trade_signal tool. `reason` = brief summary including the score (e.g., "Score 9/10: 1H BOS, sweep, deep discount, 15m FVG").
"""


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
                "confidence": {"type": "string", "enum": ["high"]}, # Enforce High
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
        logger.error("DEEPSEEK_API_KEY missing")
        return None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Market data for this batch:\n{payload_json}"},
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
            logger.warning("DeepSeek err (%d/%d): %s — retry %ds",
                           attempt + 1, _API_RETRIES, exc, wait)
            await asyncio.sleep(wait)

    if response is None or not response.choices:
        logger.error("DeepSeek returned empty")
        return None

    msg = response.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning:
        logger.info("🧠 Reasoning (%d chars): %s …",
                    len(reasoning), reasoning[:500].replace("\n", " "))
        _save_reasoning(reasoning)

    try:
        usage = response.usage
        if usage:
            logger.info("📊 Tokens — prompt: %d, completion: %d",
                        usage.prompt_tokens or 0,
                        usage.completion_tokens or 0)
    except Exception:
        pass

    if not msg.tool_calls:
        content = (msg.content or "").strip()
        if content:
            logger.info("Agent reply: %s", content[:200])
        else:
            logger.info("Agent NO_SETUP")
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

    logger.info("🚀 Signal: %s %s (HIGH)",
                args["direction"], args["symbol"])

    if await send_signal(args):
        logger.info("Signal broadcast for %s", args["symbol"])
        return args
    logger.error("Telegram send failed")
    return None
