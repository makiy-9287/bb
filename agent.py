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
ICT / SMC / Liquidity + Support resistance strategies. analyze deeply this all coins 3 timeframes . 

## INPUT FORMAT
You receive ALL 40 coins in ONE payload.
Each symbol has 3 timeframes:
  4h  → last 150 candles
  1h  → last 100 candles
  15m → last 100 candles

Per timeframe fields:
  rsi           = latest RSI(14)
  ma7,ma25,ma99 = latest simple MAs
  atr           = latest ATR(14)
  last_close    = most recent close price
  candles       = array of [open, high, low, close, volume]
                  oldest first → newest last

The LAST candle in each array is the most recent CLOSED candle.

*** Use SMC / ICT + liqudity strategy all analyze keys to analyze this coins charts. detect candlestick pattern and chart pattern also. get higher accuracy setup for LONG or SHORT with high probability TP levels and right point SL . 

## HOW TO READ THE CANDLES
- HIGH/LOW define structure. CLOSE defines direction.
- Bodies vs wicks tell you intent. Big wick + small body = rejection.
- Volume spikes = institutional activity.
- Its a binance fetched OHLCV raw data. structures full candlestick raw data to clear analyze method and deep analyze all 40 coin to get best high accuracy signals. 

##  Must analyze this all keys
 * Major support resistance or OB ,BB ( supply demand ) 
 * Swing Highs / Lows
 *  BOS (Break of Structure)
 *  CHoCH (Change of Character)
 *  Order Block (OB)
 * Fair Value Gap (FVG)
 * Breaker Block (BB)
 * Liquidity Sweep
 *  Premium / Discount with fibonachi ( set golden level to entry zone calculations 
 * POC 

## ENTRY — ALL 5 MUST ALIGN (else NO_SETUP)
1. BOS or CHoCH aligned across 2+ timeframes ( BOS Body ≥ 1.5 × ATR ) 
2. Price in DISCOUNT (LONG) or PREMIUM (SHORT)
3. OB, FVG, or Breaker Block confluence nearby . Some major Support resistance or liqudity levels are nearby
4. Recent liquidity sweep . • Liquidity Sweep Rule: Check if the current 15M candle's Low went below the recent 1H Swing Low (or Daily Low), but the candle Closed back above that Swing Low level.
• Wick Detection: Calculate the lower wick size: Lower Wick = Minimum(Open, Close) - Low. If the lower wick is significantly larger than the candle body, and it happened below a key structural low, confirm it as a Liquidity Sweep.
5. RSI supportive and POC RVOL > 1, POC confluence , HTF bias alignment 
** Also analyze more conformation like POC , volume , liqudity , support resistance etc. i need best of best setup for get higher winrate.

## SL / TP / ENTRY — USE EXACT PRICES FROM RAW CANDLES



### SL placement
- Place SL ATR-based SL (0.3-0.5 × 15m ATR) beyond the sweep low/high OR the OB edge OR the
  swing point that would invalidate the setup
- Use EXACT raw high/low values from the candle arrays
- If the structure SL is tighter than the minimum floor, use the floor

### TP placement
- TP1: nearest opposing liquidity OR 1R
- TP2: next swing extreme OR 2R
- TP3: next major liquidity OR 3R
- Tps arw must be a high accuracy liqudity pools. 
## RULES
- Both LONG and SHORT allowed
- Emit AT MOST accuracy signal for the entire payload 
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
