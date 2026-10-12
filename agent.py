"""
DeepSeek agent — strict 5-confirmation SMC/ICT filter with
extended liquidity analysis.
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
  4. Liquidity sweep confirmation (see swB / swS below)
  5. RSI supportive (not overbought for LONG / oversold for SHORT)

If ANY confirmation is missing, reply "NO_SETUP". Do NOT force trades.

## LIQUIDITY SWEEP — CRITICAL for confirmation #4
The payload gives you confirmed sweeps:
  swB = most recent bullish sweep (wick below prior swing low, closed back above)
        → {lvl: swept level, pen: wick penetration in ATR multiples, bar: candles ago}
  swS = most recent bearish sweep (wick above prior swing high, closed back below)

VALIDATION rules for confirmation #4:
  - For LONG: require swB present AND bar <= 8 (swept within last ~2 hours on 15m)
  - For SHORT: require swS present AND bar <= 8
  - Ignore sweeps older than 8 bars — they are stale.
  - Higher pen (0.5+) indicates stronger institutional rejection.

## SL/TP Rules (CRITICAL)
SL placement (in priority order):
  1. For LONG: below the nearest structure level that invalidates the setup
     (recent swing low, OB bottom, BB bottom, or swept low)
  2. For SHORT: above the nearest structure level
  3. Add buffer = 0.2 × ATR beyond the structure
  4. Absolute minimum SL distance from entry:
     BTC / ETH   : 0.6% of entry
     Major alts  : 0.9% (SOL, BNB, XRP, DOGE, ADA, AVAX, LINK, TON)
     Other alts  : 1.2%

TP placement:
  TP1: nearest opposing liquidity level OR 1R (whichever is FARTHER)
  TP2: next S/R cluster OR 2R (whichever is FARTHER)
  TP3: opposite-side liquidity (liqBS for LONG) OR 3R (whichever is FARTHER)

IMPORTANT: The bot auto-moves SL to breakeven after TP1 and to TP1 after
TP2. So TP1 should be positioned where partial profit is meaningful, and
TP3 should be a realistic final target.

## Confidence Tier
  - All 5 aligned + 4H trend strongly directional   → "high"
  - All 5 aligned + 4H trend moderate                → "medium"
  - All 5 aligned but 15m entry imperfect            → "low"

## Field Reference
bb / bbT / bbB    = Breaker Block (former OB broken opposite direction)
obL / fvgL        = top 3 recent Order Blocks / FVGs
obM / fvgM        = mitigation status (yes = price already tapped)
liqBS / liqSS     = buy-side / sell-side liquidity levels
sr                = top 5 S/R clusters {p: price, t: type, n: tests, dist: %}
rsiDiv            = RSI divergence (bull/bear)
rv                = relative volume vs 20-period avg
vah / val         = Value Area High/Low (70% of volume)
oteL / oteS       = Optimal Trade Entry zones
swB / swS         = liquidity sweeps (see above)
eqH / eqL         = equal highs / equal lows (high-prob liquidity zones)
ses               = session H/L from previous UTC day
                    {asia:{h,l}, london:{h,l}, ny:{h,l}}
dispB / dispS     = displacement candles {body: ATR multiple, bar: candles ago}
sh2 / sl2         = second-to-last swing high / low

## Discipline
Quality over quantity. A cycle with no signal is perfectly acceptable.
Emit reason as a concise confluence summary (max 120 chars)."""


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
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
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
            logger.warning("DeepSeek err (attempt %d/%d): %s — retry %ds",
                           attempt + 1, _API_RETRIES, exc, wait)
            await asyncio.sleep(wait)

    if response is None or not response.choices:
        logger.error("DeepSeek returned empty")
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
            logger.info("Agent NO_SETUP (reasoning=%d chars)",
                        len(reasoning) if reasoning else 0)
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
