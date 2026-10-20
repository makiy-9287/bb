"""
v4.5 Agent — Strict 10-Point Scoring & Safety Filter.
Prevents AI hallucinations and blocks low-score signals.
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
You analyze a batch of coins and score them. 

## STRICT RULE
You ONLY emit a signal if the score is 8, 9, or 10 (out of 10). 
If the best setup in the batch scores 7 or below, you MUST NOT call the tool. You must reply with the exact text: "NO_SETUP".

## DATA FORMAT
You get up to 5 coins per batch. Each coin has:
- 4h (150), 1h (100), 15m (100) candles: [high, low, close, volume].

## THE 10-POINT SCORING SYSTEM
1. **Market Structure (2 pts):**
   - 2 pts: Clear BOS/CHoCH on BOTH 15m and 1H in the same direction.
   - 1 pt: Clear BOS/CHoCH on 15m only, or conflicting 1H structure.
   - 0 pts: Choppy or against 1H structure.

2. **Liquidity Sweep (2 pts):**
   - 2 pts: Recent sweep of a MAJOR liquidity level (e.g., 1H swing high/low) followed by strong rejection and 15m CHoCH.
   - 1 pt: Sweep of a minor 15m level.
   - 0 pts: No sweep.

3. **Order Block / FVG Confluence (2 pts):**
   - 2 pts: Price tapping into a fresh, high-volume 1H or 15m Order Block (OB) or FVG with immediate rejection.
   - 1 pt: Price near an OB/FVG but hasn't shown a clear 15m rejection yet.
   - 0 pts: No OB/FVG nearby.

4. **Premium / Discount & POC (2 pts):**
   - 2 pts: Price DEEP in Discount for LONG (below 50% of 4H range) or DEEP in Premium for SHORT. AND price near POC.
   - 1 pt: Price in Discount/Premium, but not near POC.
   - 0 pts: Price in Equilibrium or wrong zone.

5. **Momentum & Indicators (2 pts):**
   - 2 pts: RSI supportive (<40 for LONG, >60 for SHORT). MA(7) crossing MA(25) in trade direction. Volume spike confirms.
   - 1 pt: Mixed indicators.
   - 0 pts: RSI overbought/oversold against you, or MAs against you.

## HOW TO CALCULATE SL / TP
- **SL:** 3-5 ticks beyond the structural invalidation point (swept low/high or OB edge) + 0.2x ATR buffer.
- **TP1:** Nearest 1H opposing liquidity (1R-1.5R).
- **TP2:** Next 1H swing extreme (2R).
- **TP3:** 3R.

## OUTPUT
If Score >= 8, call `send_trade_signal` with confidence="high" and the exact score.
If Score < 8, reply exactly "NO_SETUP". Do not force trades.
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
                "reason":     {"type": "string", "description": "Brief summary of the setup and score."},
            },
            "required": ["symbol", "direction", "entry", "sl",
                         "tp1", "tp2", "tp3", "confidence", "score", "reason"],
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
                    len(reasoning), reasoning[:300].replace("\n", " "))
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
            logger.info("Agent NO_SETUP (Empty content)")
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

    # ── 🚨 SAFETY CHECK: AI Hallucination Filter 🚨 ──────────────────
    score = args.get("score", 0)
    reason_text = str(args.get("reason", "")).upper()
    
    # Block if Score < 8
    if score < 8:
        logger.warning(f"🚫 BLOCKED: AI emitted a low score of {score}/10. Signal rejected.")
        return None
        
    # Block if AI wrote "INVALID" or "DO NOT EXECUTE" in the reason
    if "INVALID" in reason_text or "DO NOT EXECUTE" in reason_text:
        logger.warning(f"🚫 BLOCKED: AI reason contains warning text. Signal rejected.")
        return None

    # ── Passed checks, proceed ──────────────────────────────────────
    logger.info("🚀 Signal: %s %s | Score: %d/10 | Confidence: %s",
                args["direction"], args["symbol"], score, args.get("confidence"))

    if await send_signal(args):
        logger.info("✅ Signal broadcast for %s", args["symbol"])
        return args
    else:
        logger.error("❌ Telegram send failed")
        return None
