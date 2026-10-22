"""
OpenRouter agent pool — 15 parallel Nemotron 3 Ultra calls.
Fixed:
- Added required OpenRouter headers (HTTP-Referer, X-Title)
- Added robust 429 Rate Limit handling with exponential backoff
- Added validation for 401 Unauthorized keys (logs exact key index)
- Improved group splitting (exactly 4 coins per group when possible)
- Improved streaming tool call handling
"""
import asyncio
import json
import logging
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from config import (
    OPENROUTER_API_KEYS, OPENROUTER_BASE_URL, OPENROUTER_MODEL,
    REASONING_EFFORT, TOP_COINS_LIMIT
)
from telegram_sender import send_signal

logger = logging.getLogger(__name__)

_CLIENTS: List[AsyncOpenAI] = []
_API_TIMEOUT = 180
_RETRIES     = 3
_MAX_CONCURRENT = 5  # Increased slightly for better throughput


def _get_clients() -> List[AsyncOpenAI]:
    global _CLIENTS
    if not _CLIENTS:
        _CLIENTS = [
            AsyncOpenAI(
                api_key=k,
                base_url=OPENROUTER_BASE_URL,
                # OpenRouter REQUIRES these headers for attribution and to avoid 401
                default_headers={
                    "HTTP-Referer": "https://github.com/yourusername/ai-trading-agent",
                    "X-Title": "AI Trading Agent v6 Ultra",
                }
            )
            for k in OPENROUTER_API_KEYS
        ]
    return _CLIENTS


SYSTEM_PROMPT = """You are an elite crypto futures day trader specialising in SMC/ICT,
liquidity, support/resistance, and premium/discount strategies.

You receive a compact JSON payload for a group of coins.
Each coin has 1D, 4H, 1H, and 15m sections.

## Your Workflow (STRICT ORDER)
1. **1D** — identify major bias (bullish / bearish / range).
2. **4H** — refine bias and locate HTF OB/FVG/breaker blocks.
3. **1H** — MAIN BIAS. Confirm BOS/CHoCH and premium/discount.
4. **15m** — SNIPER ENTRY ONLY. Find the precise OB/FVG/POC zone.

## Entry Logic
- Find the highest-probability **POC zone** (pocH / pocL).
- If CMP (last_close) is INSIDE that POC zone AND all 5 confirmations align:
    entry_type = "CMP", entry = CMP.
- Else, use a **LIMIT** order at the nearest high-quality OB/FVG zone
  that overlaps a strong POC. Only if the zone is within a reasonable
  distance (≤ 2.5% from CMP). entry_type = "LIMIT",
  entry = POC inside that zone, zone_high/zone_low = OB/FVG bounds.
- If the nearest valid zone is farther than 2.5%, reply NO_SETUP.

## MANDATORY 5-Confirmation Rule
A signal is ONLY valid when ALL 5 align:
  1. Market structure (BOS/CHoCH) aligned across 2+ timeframes
  2. Price in discount (LONG) / premium (SHORT) OR at a high-prob POC
  3. Order Block OR FVG confluence nearby
  4. Recent liquidity sweep (swp) confirmed
  5. RSI supportive (not overbought for LONG / oversold for SHORT)

## SL Rules
- LONG: SL = below the sweep wick (swp.cl or swing_low) − 0.5×ATR
- SHORT: SL = above the sweep wick (swp.ch or swing_high) + 0.5×ATR
- SL must be at least 0.8×ATR away from entry.
- Include the sweep level in `sweep_level`.

## TP Rules (day-trade — must be reachable today)
- TP1: nearest liquidity pool (liqBS for LONG / liqSS for SHORT)
  OR nearest S/R cluster (sr), whichever is closer.
- TP2: next liquidity/S/R level OR VAH/VAL.
- TP3: opposite-side liquidity OR range_high/range_low.
- Ensure TP1 < TP2 < TP3 for LONG (reverse for SHORT).
- rr_ratio = (TP1 − entry) / (entry − SL) for LONG; reverse for SHORT.
  Minimum acceptable rr_ratio = 1.5.

## Confidence
- All 5 + HTF strongly directional → "high"
- All 5 + HTF moderate → "medium"
- All 5 + 15m entry slightly imperfect → "low"

## Output
If a valid setup exists, call `send_trade_signal` with exact numbers.
If not, reply exactly "NO_SETUP". Do not force trades.

## Field Reference
pocH/pocL  = POC zone high/low
inPoc      = true if CMP is inside POC zone
pocD       = % distance from CMP to POC
swp        = recent sweep {t: bear/bull, l: level, ch/cl: wick}
eqh/eql    = equal highs/lows (liquidity pools)
liqBS/liqSS = buy-side / sell-side liquidity
sr         = top 5 S/R clusters
obL/fvgL   = top 3 OBs/FVGs
obM/fvgM   = mitigation status
rsiDiv     = divergence
rv         = relative volume
vah/val    = value area
oteL/oteS  = optimal trade entry zones
sh2/sl2    = second-to-last swings"""


TOOLS = [{
    "type": "function",
    "function": {
        "name": "send_trade_signal",
        "description": "Send a confirmed SMC/ICT day-trade signal.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "direction": {"type": "string", "enum": ["LONG", "SHORT"]},
                "entry_type": {"type": "string", "enum": ["CMP", "LIMIT"]},
                "entry": {"type": "number"},
                "zone_high": {"type": "number"},
                "zone_low": {"type": "number"},
                "sl": {"type": "number"},
                "tp1": {"type": "number"},
                "tp2": {"type": "number"},
                "tp3": {"type": "number"},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                "reason": {"type": "string"},
                "poc_high": {"type": "number"},
                "poc_low": {"type": "number"},
                "sweep_level": {"type": "number"},
                "rr_ratio": {"type": "number"},
            },
            "required": [
                "symbol", "direction", "entry_type", "entry",
                "sl", "tp1", "tp2", "tp3", "confidence",
                "reason", "rr_ratio",
            ],
        },
    },
}]


def _save_reasoning(tag: str, reasoning: str):
    try:
        os.makedirs("debug_reasoning", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        with open(f"debug_reasoning/{ts}_{tag}.txt", "w", encoding="utf-8") as f:
            f.write(reasoning)
    except Exception as e:
        logger.warning("Could not save reasoning: %s", e)


async def _stream_chat(client, messages) -> tuple:
    """Stream a completion with proper reasoning and tool call handling."""
    stream = await client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=16000,
        stream=True,
        # OpenRouter requires reasoning to be passed inside extra_body
        extra_body={
            "reasoning": {
                "effort": REASONING_EFFORT,
                "enabled": True
            }
        },
    )

    content_parts, reasoning_parts = [], []
    tool_calls: Dict[int, Dict[str, str]] = {}

    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta is None:
            continue

        if delta.content:
            content_parts.append(delta.content)

        # OpenRouter sends reasoning in delta.reasoning or delta.model_extra
        r = getattr(delta, "reasoning", None)
        if r is None and getattr(delta, "model_extra", None):
            r = delta.model_extra.get("reasoning")
        if r:
            reasoning_parts.append(r)

        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index if tc.index is not None else 0
                if idx not in tool_calls:
                    tool_calls[idx] = {"name": "", "arguments": ""}
                if tc.function:
                    if tc.function.name:
                        tool_calls[idx]["name"] += tc.function.name
                    if tc.function.arguments:
                        tool_calls[idx]["arguments"] += tc.function.arguments

    return ("".join(content_parts), "".join(reasoning_parts), tool_calls)


async def _analyze_group(client, key_idx: int, group_id: int, symbols: List[str], payload: str) -> Optional[Dict[str, Any]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Group {group_id} — analyse these {len(symbols)} coins:\n{payload}"},
    ]

    for attempt in range(_RETRIES):
        try:
            content, reasoning, tool_calls = await asyncio.wait_for(
                _stream_chat(client, messages), timeout=_API_TIMEOUT,
            )
            break
        except Exception as exc:
            # 1. Handle 401 Unauthorized (Bad Key)
            if hasattr(exc, "status_code") and exc.status_code == 401:
                logger.error("❌ Group %d: API Key %d is INVALID (401). Skipping.", group_id, key_idx)
                return None
                
            # 2. Handle 429 Rate Limit (OpenRouter free tier limits)
            if hasattr(exc, "status_code") and exc.status_code == 429:
                wait = (2 ** attempt) * 3 + random.uniform(1, 3)
                logger.warning("⏳ Group %d: Rate limited (429). Retry %d/%d in %.1fs...", group_id, attempt+1, _RETRIES, wait)
                await asyncio.sleep(wait)
                continue

            # 3. Handle other errors
            if attempt == _RETRIES - 1:
                logger.error("Group %d failed after %d attempts: %s", group_id, _RETRIES, exc)
                return None
            
            wait = 2 ** attempt
            logger.warning("Group %d error (attempt %d): %s — retry %ds", group_id, attempt + 1, exc, wait)
            await asyncio.sleep(wait)

    if reasoning:
        logger.info("🧠 Group %d reasoning (%d chars): %s …", group_id, len(reasoning), reasoning[:300].replace("\n", " "))
        _save_reasoning(f"group{group_id}", reasoning)

    if not tool_calls:
        content = (content or "").strip()
        if content:
            logger.info("Group %d reply: %s", group_id, content[:150])
        else:
            logger.info("Group %d → NO_SETUP", group_id)
        return None

    for idx, tc in tool_calls.items():
        if tc["name"] != "send_trade_signal":
            continue
        args_str = tc["arguments"].strip()
        if not args_str:
            continue
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            continue

        args["_group_id"] = group_id
        logger.info("🚀 Group %d signal: %s %s (%s, conf=%s, RR=%.2f)",
                    group_id, args.get("direction"), args.get("symbol"),
                    args.get("entry_type"), args.get("confidence"), args.get("rr_ratio", 0))
        return args

    return None


def _rank_signals(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    conf_score = {"high": 3, "medium": 2, "low": 1}
    by_symbol: Dict[str, Dict] = {}
    for s in signals:
        sym = s["symbol"]
        if sym not in by_symbol:
            by_symbol[sym] = s
        else:
            cur = by_symbol[sym]
            if (conf_score.get(s["confidence"], 0) > conf_score.get(cur["confidence"], 0)):
                by_symbol[sym] = s

    return sorted(
        by_symbol.values(),
        key=lambda x: (conf_score.get(x.get("confidence"), 0), float(x.get("rr_ratio", 0))),
        reverse=True,
    )


async def run_agent_batch(groups: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Run groups in parallel with throttling to avoid 429s."""
    clients = _get_clients()
    if not clients:
        return None

    n = min(len(clients), len(groups))
    sem = asyncio.Semaphore(_MAX_CONCURRENT)

    async def _throttled_task(idx: int):
        # Stagger launch times to avoid burst rate limits
        await asyncio.sleep(idx * 1.5) 
        async with sem:
            return await _analyze_group(
                clients[idx], idx + 1, groups[idx]["id"], 
                groups[idx]["symbols"], groups[idx]["payload"]
            )

    tasks = [_throttled_task(i) for i in range(n)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_signals = []
    for r in results:
        if isinstance(r, Exception):
            logger.error("Task crashed: %s", r)
            continue
        if r:
            all_signals.append(r)

    if not all_signals:
        logger.info("All groups → NO_SETUP (or failed)")
        return None

    ranked = _rank_signals(all_signals)
    best = ranked[0]

    logger.info("🏆 Best signal: %s %s (conf=%s, RR=%.2f) from group %d",
                best["direction"], best["symbol"], best.get("confidence"),
                best.get("rr_ratio", 0), best.get("_group_id", 0))

    if await send_signal(best):
        return best
    return None                    
