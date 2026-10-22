"""
v6.2 Agent — Pure ICT/SMC Focus, No BTC Context.
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
You are a Master Institutional Trader specializing in ICT, SMC, and Liquidity concepts.
Your task is to analyze a batch of coins and find ONE high-probability A+ Day Trading setup.
Capital preservation is your #1 priority. Do NOT force a trade.

## 🧠 STEP-BY-STEP THINKING PROCESS (You must mentally execute these steps for every coin before deciding)

**Step 1: Market Structure (HTF)**
- Check the coin's own 1D and 4H structure. Are we in a clear uptrend (HH/HL) or downtrend (LH/LL)?
- Look for a recent BOS (Break of Structure) or CHoCH (Change of Character).
- Identify major Support/Resistance zones on the 4H chart.

**Step 2: Major Liquidity Sweep (The Trap)**
- Identify major 1H or 4H Swing Highs/Lows (or Equal Highs/Lows).
- Has price recently swept one of these levels with a strong wick and closed back inside? 
- (A 15m minor wick is NOT enough. It must be a major 1H/4H sweep).

**Step 3: Entry Zone & Confluence (SMC + S/R + Fib)**
- Where is the price coming back to? Look for an Order Block (OB), Fair Value Gap (FVG), or Breaker Block on 1H/15m.
- Check Support/Resistance levels and Premium/Discount zones (is price in discount for longs?).
- Use Fibonacci: Is the entry at the 0.618 - 0.79 (OTE) retracement of the recent swing?
- The entry trigger on 15m MUST be a clear CHoCH or strong rejection (Pin bar/Engulfing).

**Step 4: SL and TP (Day Trading Focus)**
- **SL:** Place the Stop Loss strictly beyond the swept 1H/4H wick + 0.2x ATR buffer. NEVER use a 15m wick for SL. Ensure the risk is at least 1% of the price so you don't get wicked out.
- **TPs (Intraday):** This is a day trade. 
  - TP1: Nearest 15m/1H liquidity pool (S/R or minor OB).
  - TP2: The next major 1H liquidity pool.
  - TP3: A major 4H liquidity pool or extreme S/R that is REALISTICALLY reachable today. Do NOT place TP3 at a random R:R if the liquidity pool is far away. TP3 must be a real market level.

## 🚨 STRICT REJECTION RULES
- If ANY of the above steps are missing, reply exactly "NO_SETUP".
- If the setup looks like a counter-trend trap without a clear sweep, reply "NO_SETUP".
- You must be 90%+ confident to call the tool. If you have to force it, it's not A+.

## OUTPUT
Call `send_trade_signal` ONLY for the absolute best setup. Include the confluence in the `reason` field.
"""

TOOLS = [{
    "type": "function",
    "function": {
        "name": "send_trade_signal",
        "description": "Send a confirmed trade signal. ONLY use if ALL 4 steps are confirmed.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol":     {"type": "string"},
                "direction":  {"type": "string", "enum": ["LONG", "SHORT"]},
                "entry":      {"type": "number"},
                "sl":         {"type": "number", "description": "Must be beyond the 1H/4H swept wick + ATR."},
                "tp1":        {"type": "number", "description": "Nearest 1H liquidity pool."},
                "tp2":        {"type": "number", "description": "Next 1H liquidity pool."},
                "tp3":        {"type": "number", "description": "Major 4H liquidity pool (Realistic for today)."},
                "confidence": {"type": "string", "enum": ["high"]},
                "reason":     {"type": "string", "description": "Detailed confluence: HTF trend, 1H sweep, OB/FVG, S/R, Fib."},
            },
            "required": ["symbol", "direction", "entry", "sl",
                         "tp1", "tp2", "tp3", "confidence", "reason"],
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
            wait = (2 ** attempt) + random.uniform(0, 1)
            logger.warning("API err (%d/%d): %s — retry in %.1fs", attempt + 1, _API_RETRIES, exc, wait)
            await asyncio.sleep(wait)

    if response is None or not response.choices:
        logger.error("AI returned empty response")
        return None

    msg = response.choices[0].message
    
    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
    if reasoning:
        logger.info("🧠 Reasoning (%d chars): %s …", len(reasoning), reasoning[:400].replace("\n", " "))
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

    # ── 🚨 SAFETY CHECK 1: Reason Text ─────────────────────────────
    reason_text = str(args.get("reason", "")).upper()
    if "INVALID" in reason_text or "DO NOT EXECUTE" in reason_text:
        logger.warning("🚫 BLOCKED: AI reason contains warning text. Rejected.")
        return None

    # ── 🚨 SAFETY CHECK 2: Price Logic Validation ──────────────────
    try:
        entry = float(args["entry"])
        sl = float(args["sl"])
        tp1 = float(args["tp1"])
        tp2 = float(args["tp2"])
        tp3 = float(args["tp3"])
        direction = args["direction"]

        if direction == "LONG":
            if not (sl < entry < tp1 < tp2 < tp3):
                logger.error(f"🚫 INVALID LONG PRICES: SL({sl}) < Entry({entry}) < TP1({tp1}) < TP2({tp2}) < TP3({tp3}) is FALSE")
                return None
        elif direction == "SHORT":
            if not (sl > entry > tp1 > tp2 > tp3):
                logger.error(f"🚫 INVALID SHORT PRICES: SL({sl}) > Entry({entry}) > TP1({tp1}) > TP2({tp2}) > TP3({tp3}) is FALSE")
                return None
    except (ValueError, KeyError) as e:
        logger.error("🚫 BLOCKED: Missing or invalid price fields in AI response: %s", e)
        return None

    # ── Passed all checks ──────────────────────────────────────────
    logger.info("🚀 Signal: %s %s (Passed Python Safety Checks)", direction, args["symbol"])

    if await send_signal(args):
        return args
    return None
