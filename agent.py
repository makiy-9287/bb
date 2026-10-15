"""
v4.1 Agent — pure raw-candle analysis, single request.
Prompt tuned for reading OHLCV arrays + SL minimums by coin tier.
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


SYSTEM_PROMPT = """You are an elite Binance Futures day trader. You trade ONLY with
ICT / SMC / Liquidity strategies. Nothing else.

## INPUT FORMAT
You receive ALL 60 coins in ONE payload.
Each symbol has 3 timeframes:
  4h  → last 300candles
  1h  → last 250 candles
  15m → last 100 candles

Per timeframe fields:
  rsi           = latest RSI(14)
  ma7,ma25,ma99 = latest simple MAs
  atr           = latest ATR(14)
  last_close    = most recent close price
  candles       = array of [open, high, low, close, volume]
                  oldest first → newest last

The LAST candle in each array is the most recent CLOSED candle.

## HOW TO READ THE CANDLES
- HIGH/LOW define structure. CLOSE defines direction.
- Bodies vs wicks tell you intent. Big wick + small body = rejection.
- Volume spikes = institutional activity.

## HOW TO DETECT EACH CONCEPT FROM RAW CANDLES

### Swing Highs / Lows
A swing high = candle whose HIGH is higher than the 2 candles before
and 2 candles after it. Swing low = mirror.

### BOS (Break of Structure)
Bullish BOS = a candle CLOSES above the most recent swing high.
Bearish BOS = a candle CLOSES below the most recent swing low.

### CHoCH (Change of Character)
Bullish CHoCH = after a series of lower lows, price CLOSES above the
last swing high (first sign of reversal up).
Bearish CHoCH = after higher highs, price CLOSES below last swing low.

### Order Block (OB)
Bullish OB = the last BEARISH candle before a strong bullish move that
breaks structure. Mark its HIGH and LOW.
Bearish OB = the last BULLISH candle before a strong bearish move that
breaks structure.

### Fair Value Gap (FVG)
Bullish FVG = 3-candle pattern where candle[i-1].high < candle[i+1].low.
The gap = [candle[i-1].high, candle[i+1].low]. Price often returns to fill.
Bearish FVG = candle[i-1].low > candle[i+1].high.

### Breaker Block (BB)
A former OB that price has broken THROUGH in the opposite direction.
It now acts as support/resistance on retest.

### Liquidity Sweep
A candle WICKS beyond a prior swing high or low but CLOSES back inside.
- Bullish sweep: wicks BELOW a swing low, closes ABOVE that low
- Bearish sweep: wicks ABOVE a swing high, closes BELOW that high
The swept level is your key reference.

### Premium / Discount
Take the range of the last ~50 candles (highest high - lowest low).
  Above 50% midpoint = PREMIUM (look for SHORT)
  Below 50% midpoint = DISCOUNT (look for LONG)
Optimal Trade Entry (OTE) = 0.618 to 0.79 retracement of the range.

## ENTRY — ALL 5 MUST ALIGN (else NO_SETUP)
1. BOS or CHoCH aligned across 2+ timeframes
2. Price in DISCOUNT (LONG) or PREMIUM (SHORT)
3. OB, FVG, or Breaker Block confluence nearby
4. Recent liquidity sweep (within last ~8 candles of that TF)
5. RSI supportive (not overbought for LONG / oversold for SHORT)

## SL / TP — USE EXACT PRICES FROM RAW CANDLES



### SL placement
- Place SL 3-5 ticks beyond the sweep low/high OR the OB edge OR the
  swing point that would invalidate the setup
- Use EXACT raw high/low values from the candle arrays
- If the structure SL is tighter than the minimum floor, use the floor

### TP placement
- TP1: nearest opposing liquidity OR 1R, whichever is FARTHER
- TP2: next swing extreme OR 2R
- TP3: next major liquidity OR 3R

## RULES
- Both LONG and SHORT allowed
- Emit AT MOST ONE signal for the entire payload
- If no valid setup exists: reply exactly NO_SETUP. Do not force trades.
- Never invent candles that are not in the data.

## OUTPUT
Use the send_trade_signal tool. `reason` = concise ICT summary (max 120 chars)."""


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


def _save_reasoning(reasoning: str):
    try:
        os.makedirs("debug_reasoning", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        with open(f"debug_reasoning/{ts}.txt", "w", encoding="utf-8") as f:
            f.write(reasoning)
    except Exception as e:
        logger.warning("Could not save reasoning: %s", e)


async def run_agent(payload_json: str) -> Optional[Dict[str, Any]]:
    """Single request — all coins in one payload."""
    if not DEEPSEEK_API_KEY:
        logger.error("DEEPSEEK_API_KEY missing")
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
            logger.info("📊 Tokens — prompt: %d, completion: %d, total: %d",
                        usage.prompt_tokens or 0,
                        usage.completion_tokens or 0,
                        usage.total_tokens or 0)
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

    logger.info("🚀 Signal: %s %s (%s)",
                args["direction"], args["symbol"], args.get("confidence"))

    if await send_signal(args):
        logger.info("Signal broadcast for %s", args["symbol"])
        return args
    logger.error("Telegram send failed")
    return None
