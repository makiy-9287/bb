"""
v5 Agent — OpenRouter (Nemotron 3 Ultra) with dynamic API key.
"""
import asyncio
import json
import logging
import os
import random
from datetime import datetime
from typing import Any, Dict, Optional

from openai import AsyncOpenAI

from config import (
    OPENROUTER_BASE_URL, OPENROUTER_MODEL,
    REASONING_EFFORT,
)
from telegram_sender import send_signal

logger = logging.getLogger(__name__)
_API_RETRIES = 3

SYSTEM_PROMPT = """
You are an elite ICT/SMC trader managing a hedge fund.
Your ONLY job is to find the absolute BEST setups. Capital preservation is priority #1.
You analyze a batch of 10 coins and score them.

## STRICT RULE
You ONLY emit a signal if the score is 8, 9, or 10 (out of 10). 
If the best setup in the batch scores 7 or below, you MUST NOT call the tool. You must reply with the exact text: "NO_SETUP".

## DATA FORMAT
You get up to 10 coins per batch. Each coin has:
- 1d (50), 4h (150), 1h (100), 15m (100) candles: [high, low, close, volume].

## THE 10-POINT SCORING SYSTEM
1. **Market Structure (2 pts):** 
   - 2 pts: Clear BOS/CHoCH on 15m AND 1H in same direction + 4H/1D Trend aligned.
   - 1 pt: 15m only, or conflicting 1H.
   - 0 pts: Choppy.

2. **Liquidity Sweep (2 pts):**
   - 2 pts: Recent sweep of MAJOR liquidity (1H swing) + strong rejection + 15m CHoCH.
   - 1 pt: Minor 15m sweep.
   - 0 pts: No sweep.

3. **OB / FVG (2 pts):**
   - 2 pts: Price tapping into fresh 1H/15m OB or FVG with immediate rejection.
   - 1 pt: Near zone, no clear rejection.
   - 0 pts: No zone.

4. **Premium / Discount & POC (2 pts):**
   - 2 pts: Deep Discount (LONG) or Premium (SHORT) + near POC/VAH/VAL.
   - 1 pt: In zone, but not near POC.
   - 0 pts: Equilibrium.

5. **Momentum (2 pts):**
   - 2 pts: RSI supportive + MAs aligned + Volume confirms.
   - 1 pt: Mixed indicators.
   - 0 pts: RSI overbought/oversold against you.

## HOW TO CALCULATE SL / TP
- **SL:** 3-5 ticks beyond swept low/high or OB edge + 0.2x ATR buffer.
- **TP1:** Nearest 1H opposing liquidity (1R-1.5R).
- **TP2:** Next 1H swing extreme (2R).
- **TP3:** 3R.

## OUTPUT
If Score >= 8, call `send_trade_signal` with confidence="high".
If Score < 8, reply exactly "NO_SETUP".
"""

TOOLS = [{
    "type": "function",
    "function": {
        "name": "send_trade_signal",
        "description": "Send a confirmed trade signal to Telegram. ONLY use if score >= 8.",
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
                "confidence": {"type": "string", "enum": ["high"]},
                "score":      {"type": "integer", "description": "Setup score out of 10. Must be 8, 9, or 10."},
                "reason":     {"type": "string"},
            },
            "required": ["symbol", "direction", "entry", "sl",
                         "tp1", "tp2", "tp3", "confidence", "score", "reason"],
        },
    },
}]

def _build_request_kwargs() -> dict:
    kwargs: Dict[str, Any] = {
        "model": OPENROUTER_MODEL,
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": 16000,
    }
    # Nemotron uses standard OpenAI reasoning format
    kwargs["reasoning_effort"] = REASONING_EFFORT
    kwargs["extra_body"] = {
        "reasoning": {"effort": REASONING_EFFORT},
        "include_reasoning": True
    }
    return kwargs

def _save_reasoning(reasoning: str):
    try:
        os.makedirs("debug_reasoning", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        with open(f"debug_reasoning/{ts}.txt", "w", encoding="utf-8") as f:
            f.write(reasoning)
    except Exception:
        pass

async def run_agent(payload_json: str, api_key: str) -> Optional[Dict[str, Any]]:
    if not api_key:
        logger.error("API Key is empty!")
        return None

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={"HTTP-Referer": "https://github.com/makiy-9287/bb", "X-Title": "SMC Trading Bot"}
    )

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
                logger.error("Agent API failed after %d attempts: %s", _API_RETRIES, exc)
                return None
            
            # Exponential backoff with jitter for rate limits
            wait = (2 ** attempt) + random.uniform(0, 1)
            logger.warning("API err (%d/%d): %s — retry in %.1fs", attempt + 1, _API_RETRIES, exc, wait)
            await asyncio.sleep(wait)

    if response is None or not response.choices:
        logger.error("AI returned empty response")
        return None

    msg = response.choices[0].message
    
    # Extract reasoning (OpenRouter sends this differently sometimes)
    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
    if reasoning:
        logger.info("🧠 Reasoning (%d chars): %s …", len(reasoning), reasoning[:300].replace("\n", " "))
        _save_reasoning(reasoning)

    if not msg.tool_calls:
        content = (msg.content or "").strip()
        logger.info("Agent reply: %s", content[:150] if content else "NO_SETUP")
        return None

    args_str = msg.tool_calls[0].function.arguments or ""
    if not args_str:
        return None

    try:
        args = json.loads(args_str)
    except json.JSONDecodeError:
        logger.error("Bad tool args: %s", args_str)
        return None

    # ── 🚨 SAFETY CHECK ─────────────────────────────────────────────
    score = args.get("score", 0)
    reason_text = str(args.get("reason", "")).upper()
    
    if score < 8 or "INVALID" in reason_text or "DO NOT EXECUTE" in reason_text:
        logger.warning(f"🚫 BLOCKED: AI tried to send a bad signal (Score: {score}). Rejected.")
        return None

    logger.info("🚀 Signal: %s %s | Score: %d/10", args["direction"], args["symbol"], score)

    if await send_signal(args):
        return args
    return None
