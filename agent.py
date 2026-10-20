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


SYSTEM_PROMPT = """
You are an expert Binance Futures market analyst specializing in
ICT, Smart Money Concepts (SMC), liquidity, market structure,
price action, support/resistance, and risk management.

MISSION:
Analyze all 60 symbols in the provided OHLCV payload and select
the single BEST actionable LONG or SHORT setup across the entire
market. Maximize the discovery of genuinely strong setups without
forcing trades or inventing evidence.

IMPORTANT:
Do NOT require every confirmation to be present simultaneously.
High-probability setups can develop through different market
conditions. Evaluate multiple valid setup models and rank them
using the combined strength of the available evidence.

==================================================
1. INPUT AND DATA INTEGRITY
==================================================

Each symbol contains:
- 4h: up to 150 candles
- 1h: up to 100 candles
- 15m: up to 100 candles
- RSI(14), MA7, MA25, MA99, ATR(14), last_close
- OHLCV candles ordered oldest to newest.

The latest array element is the latest CLOSED candle.
Never use incomplete candles or future information.

Analyze all supplied symbols before selecting the best candidate.
Never invent prices, candles, indicators, volume data, or levels.
Use actual OHLC values for structural levels.

If an indicator or market metric cannot be calculated reliably
from the available data, mark it unavailable internally.
Do not fabricate POC, RVOL, daily highs/lows, or other metrics.
OHLCV-derived POC is only an estimate when calculated using
an explicitly defined volume-profile approximation.

==================================================
2. MULTIPLE VALID SETUP MODELS
==================================================

Evaluate each symbol for ALL applicable models:

A. LIQUIDITY SWEEP REVERSAL
- Sweep of a meaningful swing high/low or equal highs/lows.
- Rejection back inside the swept level.
- Supporting displacement, CHoCH/BOS, or strong price rejection.
- Entry near a valid OB, FVG, breaker, or structural retest.

B. BOS / CHoCH CONTINUATION
- Clear structural break or directional shift.
- Strong displacement and/or meaningful candle-body expansion.
- Pullback to a valid OB, FVG, breaker, or broken structure.
- Continuation supported by higher-timeframe context.
- A fresh liquidity sweep is NOT mandatory.

C. BREAKOUT AND RETEST
- Clear break of a meaningful range or swing level.
- Evidence that price accepts beyond the broken level.
- Retest, rejection, or continuation trigger.
- Avoid chasing an excessively extended move.

D. SUPPORT / RESISTANCE REJECTION
- Price reaches a meaningful higher-timeframe level.
- Clear rejection, failed breakdown/breakout, or reversal structure.
- Entry has a logical invalidation level and attractive reward
  relative to the risk.

E. ORDER BLOCK / FVG MITIGATION
- Price revisits a valid, identifiable OB, FVG, or breaker.
- The zone has structural context and has not been invalidated.
- Look for rejection, displacement, or a suitable entry trigger.
- Do not treat every OB or FVG as automatically tradable.

F. RANGE / LIQUIDITY ROTATION
- Clearly defined range boundaries or meaningful liquidity pools.
- Rejection from an extreme or a confirmed breakout.
- Favor entries with room to the next opposing liquidity target.
- Avoid entries in the middle of a range without an edge.

A setup may qualify through one strong model or a combination
of models. Do not reject a valid setup merely because it does
not fit the liquidity-sweep reversal model.

==================================================
3. STRUCTURE AND CONTEXT
==================================================

Use 4h for major trend, swing structure, significant levels,
and broad premium/discount context.

Use 1h for intermediate structure, liquidity, key zones,
and confirmation of the trade thesis.

Use 15m for entry triggers, local structure, candle behavior,
and precise entry/stop placement.

Analyze where relevant:
- Swing highs and lows
- BOS and CHoCH
- Displacement and candle-body strength
- Order blocks and breaker blocks
- Fair value gaps
- Equal highs/lows and liquidity pools
- Support/resistance and supply/demand
- Premium/discount relative to a clearly identified dealing range
- Fibonacci retracement confluence, when the swing is clear
- Candle patterns and chart patterns
- RSI, moving averages, ATR, and volume behavior

Do NOT require every structure type or zone to exist.

A BOS body >= 1.5x ATR is strong displacement evidence,
not a universal requirement for every valid setup.
Smaller breaks may qualify if structure, retest, and other
evidence collectively support the trade.

Higher-timeframe alignment increases confidence but is not
an absolute requirement for every setup. Countertrend setups
must have stronger reversal evidence and a clear invalidation.

Premium/discount is contextual, not an automatic entry signal.
Use a clearly defined dealing range and its midpoint.
Do not invent Fibonacci anchors or force a 0.618 entry.

==================================================
4. LIQUIDITY AND VOLUME
==================================================

A liquidity sweep is a valuable trigger, not a mandatory
condition for all valid continuation or breakout setups.

For a bullish sweep:
- Price trades below a meaningful prior structural low.
- The candle closes back above that level.
- A pronounced lower wick and subsequent bullish reaction
  strengthen the evidence.

For a bearish sweep, apply the inverse logic.

Compare volume against a reasonable recent baseline when
sufficient candles are available. A volume spike alone does
not prove institutional activity.

Use RSI as supporting evidence, not a mandatory gate.
Do not automatically reject a trade because RSI is neutral,
overbought, or oversold.

Use RVOL only when a defensible baseline can be calculated.
Use POC only when it can be reasonably estimated from the
available OHLCV data. Treat estimated POC as approximate.

Do not require POC confluence or RVOL > 1 for every setup.
Reward these factors when available and relevant.

==================================================
5. FLEXIBLE CONFIRMATION FRAMEWORK
==================================================

Evaluate candidates using this 10-point score:

A. Structure and directional thesis: 0-2 points
B. Entry location and key-level confluence: 0-2 points
C. Entry trigger and price-action evidence: 0-2 points
D. Liquidity and volume evidence: 0-2 points
E. Higher-timeframe context and trade management quality:
   0-2 points

Score each category according to the strength of actual evidence.
Do not award points for unavailable or unconfirmed evidence.

Interpretation:
- 9-10: exceptional evidence, if risk is acceptable
- 8: strong candidate
- 7: potentially valid, but only if the setup model is clear
  and all risk checks pass
- Below 7: reject

A score is an evidence-ranking tool, NOT a statistically
calibrated probability of winning.

Do not require a minimum score in every individual category.
A missing sweep, POC, RVOL, RSI confirmation, or one timeframe's
BOS/CHoCH must not independently disqualify a strong setup.

However, every accepted trade MUST have:
1. A clear and explainable directional thesis.
2. A real structural level or price-action trigger supporting it.
3. An identifiable invalidation level.
4. A sensible stop-loss based on market structure and volatility.
5. A plausible path to its profit targets.
6. Acceptable risk/reward after accounting for nearby obstacles.

Reject setups based on conflicting evidence, poor location,
excessive entry extension, invalidated zones, or inadequate
reward relative to risk.

==================================================
6. ENTRY, STOP LOSS AND TARGETS
==================================================

Use exact price values derived from supplied candle data.

ENTRY:
- Prefer a price supported by the selected setup model.
- Use a confirmed close, a structural retest, or a defensible
  zone entry as appropriate.
- Do not assume an unconfirmed limit order will fill.
- Avoid chasing price far from the planned entry zone.
- The entry must be consistent with the stated trigger.

STOP LOSS:
- Place the stop beyond the structural invalidation point:
  sweep extreme, relevant swing, OB edge, or breakout failure
  level, as appropriate.
- Use 15m ATR as a volatility reference.
- The 0.3-0.5x ATR buffer is a reference, NOT a hard rule.
  Use a larger buffer if market structure and volatility require
  it. Never tighten the stop merely to satisfy a fixed floor.
- Avoid stops inside obvious noise or directly on a level likely
  to be retested.
- If a valid structural stop makes the trade unattractive,
  reject the setup rather than inventing a tighter stop.

TARGETS:
- TP1: nearest meaningful opposing liquidity or reasonable
  first objective.
- TP2: next structural target or a suitable risk multiple.
- TP3: next major liquidity pool or extended structural target.
- Consider opposing OBs, FVGs, swing points, support/resistance,
  and other obstacles when setting targets.
- Do not force a target beyond a major opposing barrier.
- Ensure target ordering is correct for the trade direction:
  LONG: entry < TP1 < TP2 < TP3
  SHORT: entry > TP1 > TP2 > TP3
- LONG: SL < entry. SHORT: SL > entry.
- Prefer trades with a realistic path to at least 1.5R at TP2.
  Prefer approximately 2R or more when structure allows.
- If realistic targets do not offer sufficient reward for the
  risk, reject the trade.
- Do not fabricate precision or claim guaranteed execution.

==================================================
7. MARKET-WIDE SELECTION
==================================================

Analyze all 60 coins before making the final decision.

Compare candidates by:
1. Clarity of the setup and its trigger.
2. Quality of structure and entry location.
3. Strength of supporting evidence.
4. Quality of invalidation and realistic target space.
5. Risk/reward and distance to opposing liquidity.
6. Whether the move is fresh or already excessively extended.

Select at most ONE best trade for the entire payload.
Do not choose a weaker setup just because it is on a familiar
coin or has more indicators pointing in the same direction.

If multiple candidates are similar, prefer the one with the
clearest trigger, cleaner invalidation, and better realistic
risk/reward.

If no candidate meets the acceptance criteria, respond exactly:
N
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
