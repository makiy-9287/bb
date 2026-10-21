"""
v6 Agent — 15-Point Scoring for A+ Setups.
"""
import asyncio
import json
import logging
import os
import random
from datetime import datetime
from typing import Any, Dict, Optional

from openai import AsyncOpenAI
from config import OPENROUTER_BASE_URL, OPENROUTER_MODEL, REASONING_EFFORT
from telegram_sender import send_signal

logger = logging.getLogger(__name__)
_API_RETRIES = 3

SYSTEM_PROMPT = """
You are an elite ICT/SMC trader managing a hedge fund.
Your ONLY job is to find the absolute BEST setups. Capital preservation is priority #1.
You analyze a batch of 4 coins and score them.

## STRICT RULE
You ONLY emit a signal if the score is  14, or 15 (out of 15). 
If the best setup in the batch scores 12 or below, you MUST NOT call the tool. Reply with exact text: "NO_SETUP".

## DATA FORMAT
You get 4 coins per batch. Each coin has:
- 1d (100), 4h (250), 1h (200), 15m (150) candles: [high, low, close, volume].

## THE 15-POINT SCORING SYSTEM (Need 14+ to pass)
1. **Market Structure (SMC/ICT) (3 pts):**
   - 3 pts: Clear BOS/CHoCH on 15m AND 1H aligned with 4H/1D.
   - 1-2 pts: Minor structure breaks or mixed timeframes.
   - 0 pts: Choppy.

2. **Liquidity Sweep (2 pts):**
   - 2 pts: Recent sweep of a MAJOR 1H/4H swing high/low + strong rejection + 15m CHoCH.
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

5. **Momentum (RSI/MA/Vol) (2 pts):**
   - 2 pts: RSI supportive + MAs aligned + Volume confirms.
   - 1 pt: Mixed indicators.
   - 0 pts: Conflicting.

6. **Classical S&R / Trendlines (2 pts):**
   - 2 pts: Price is bouncing off a major 4H/1H Support/Resistance zone OR a respected multi-touch trendline.
   - 1 pt: Minor S&R or weak trendline.
   - 0 pts: No classical S&R support.

7. **Fibonacci Confluence (2 pts):**
   - 2 pts: The entry is perfectly inside the 0.618 - 0.786 Golden Zone of a major 1H/4H swing AND aligns with an OB.
   - 1 pt: Near the Fib zone.
   - 0 pts: No Fib alignment.

## STRICT SL / TP RULES (LIQUIDITY BASED)
- **SL:** Do NOT use arbitrary tight stops. Place SL exactly 3-5 ticks BEYOND the recent swept wick (the extreme low for LONG, high for SHORT) + a 0.2x ATR buffer.
- **TPs:** Do NOT use fixed 1R, 2R, 3R. Look at the chart and find the NEXT high-probability Liquidity Pools (S/R, OB, BB, Equal Highs/Lows).
  * **TP1:** Nearest 15m/1H minor liquidity pool.
  * **TP2:** Next 1H major liquidity pool or 4H OB.
  * **TP3:** 4H/1D major liquidity zone.
  * *Distance does not matter, target the actual market structure.*

## OUTPUT
If Score >= 14, call `send_trade_signal` with confidence="high".
If Score < 14, reply exactly "NO_SETUP".
"""

TOOLS = [{
    "type": "function",
    "function": {
        "name": "send_trade_signal",
        "description": "Send a confirmed trade signal. ONLY use if score >= 14.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol":     {"type": "string"},
                "direction":  {"type": "string", "enum": ["LONG", "SHORT"]},
                "entry":      {"type": "number"},
                "sl":         {"type": "number", "description": "Beyond the swept wick + ATR buffer"},
                "tp1":        {"type": "number"},
                "tp2":        {"type": "number"},
                "tp3":        {"type": "number"},
                "confidence": {"type": "string", "enum": ["high"]},
                "score":      {"type": "integer", "description": "Score out of 15. MUST be 14, or 15."},
                "reason":     {"type": "string", "description": "Internal log only, not shown to users"},
            },
            "required": ["symbol", "direction", "entry", "sl", "tp1", "tp2", "tp3", "confidence", "score", "reason"],
        },
    },
}]

def _build_request_kwargs() -> dict:
    kwargs: Dict[str, Any] = {
        "model": OPENROUTER_MODEL,
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": 16000,
        "reasoning_effort": REASONING_EFFORT,
        "extra_body": {"reasoning": {"effort": REASONING_EFFORT}, "include_reasoning": True}
    }
    return kwargs

async def run_agent(payload_json: str, api_key: str) -> Optional[Dict[str, Any]]:
    if not api_key:
        return None

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={"HTTP-Referer": "https://github.com/makiy-9287/bb", "X-Title": "SMC Bot"}
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
                logger.error("API failed: %s", exc)
                return None
            await asyncio.sleep((2 ** attempt) + random.uniform(0, 1))

    if response is None or not response.choices:
        return None

    msg = response.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
    if reasoning:
        logger.info("🧠 Reasoned (%d chars)", len(reasoning))

    if not msg.tool_calls:
        return None

    args_str = msg.tool_calls[0].function.arguments or ""
    try:
        args = json.loads(args_str)
    except:
        return None

    # ── 🚨 SAFETY CHECK (Score >= 13) ──────────────────────────────
    score = args.get("score", 0)
    reason_text = str(args.get("reason", "")).upper()
    
    if score < 13 or "INVALID" in reason_text:
        logger.warning(f"🚫 BLOCKED: Low Score ({score}/15). Rejected.")
        return None

    logger.info("🚀 Valid Signal: %s %s | Score: %d/15", args["direction"], args["symbol"], score)

    if await send_signal(args):
        return args
    return None
