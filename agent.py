"""
OpenRouter agent pool — 15 parallel Nemotron 3 Ultra calls.
Fixed: 1 request / second launch (15 keys -> 15 staggered calls, all running in parallel).
Added: retries (429/5xx/timeout), response validation, signal sanity checks.
"""
import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from config import (
    OPENROUTER_API_KEYS, OPENROUTER_BASE_URL, OPENROUTER_MODEL,
    REASONING_EFFORT,
)
from telegram_sender import send_signal
from signal_filter import refine_signal

logger = logging.getLogger(__name__)

_CLIENTS: List[AsyncOpenAI] = []
_API_TIMEOUT = 420          # Nemotron Ultra is slow (reasoning) — 180s was killing many calls
_MAX_ATTEMPTS = 3           # per group (same key)
_GROUP_DEADLINE = 900       # stop retrying a group after this many seconds
_LAUNCH_GAP_SEC = 1.0       # 1 request per second: group1 @0s, group2 @1s ... group15 @14s



def _get_clients() -> List[AsyncOpenAI]:
    global _CLIENTS
    if not _CLIENTS:
        _CLIENTS = [
            AsyncOpenAI(
                api_key=k,
                base_url=OPENROUTER_BASE_URL,
                default_headers={
                    "HTTP-Referer": "https://github.com/yourusername/ai-trading-agent",
                    "X-Title": "AI Trading Agent v6 Ultra",
                },
                max_retries=0,  # We handle retries manually
            )
            for k in OPENROUTER_API_KEYS
        ]
    return _CLIENTS


SYSTEM_PROMPT = """You are an elite crypto futures day trader (SMC/ICT, liquidity, S/R, premium/discount).
You get a compact JSON payload for a group of coins. Each coin has 1d, 4h, 1h, 15m sections.
Goal: report ONLY A+ setups. Most scans should end in NO_SETUP — that is correct behaviour.
A script re-verifies every number you send (and finalises the SL on 15m structure), so
inconsistent or forced trades are discarded. Keep your reasoning concise; do not re-derive things twice.

## Workflow (STRICT ORDER)
1. 1d  — major bias.            2. 4h — refine bias, HTF OB/FVG.
3. 1h  — MAIN BIAS: structure + premium/discount.
4. 15m — SNIPER ENTRY: exact OB/FVG/POC zone, sweep, invalidation (SL).

## Field meaning (read carefully — do not mix these up)
ls        = LATEST structure event {t: BOS|CHOCH, d: +1 bullish / -1 bearish, ago: candles}. Use this for bias.
bos/choch = last BOS / last CHOCH seen (+1 bull, -1 bear). They can be old and contradict each other; ls decides.
swp       = recent sweep. t="bear": HIGHS were swept (price rejected down) → supports SHORT; l=level, ch=wick high.
            t="bull": LOWS were swept (price rejected up) → supports LONG; l=level, cl=wick low.
pd        = premium (above equilibrium eq) / discount (below). rh/rl/eq = 50-candle range high/low/equilibrium.
pocH/pocL = POC zone; inPoc = CMP inside it; pocD = % distance CMP→POC. lc = last close (CMP).
obL/fvgL  = up to 3 OBs/FVGs [{t: bull|bear, top, bot}]. bull zone = demand (for LONG), bear zone = supply (for SHORT).
obM/fvgM  = "yes" = price already returned into/through the zone (mitigated), "no" = fresh. Prefer fresh zones.
eqh/eql   = equal highs/lows (liquidity pools) {p: price, n: touches}. liqBS/liqSS = buy-side / sell-side liquidity.
sr = S/R clusters, rsiDiv = RSI divergence, rv = relative volume, vah/val = value area, oteL/oteS = OTE zones,
sh/sl and sh2/sl2 = last / previous swing high / low, atr = ATR(14) of that timeframe.

## Direction rules (hard)
- LONG  only if: entry is in DISCOUNT (below 1h eq), 1h ls.d = +1, and at least one more of 4h/1d agrees.
- SHORT only if: entry is in PREMIUM  (above 1h eq), 1h ls.d = -1, and at least one more of 4h/1d agrees.
- NEVER long in premium, NEVER short in discount, and NEVER trade against the 1h structure.
- The sweep must match the direction: LONG needs swp.t="bull", SHORT needs swp.t="bear" (15m preferred, 1h acceptable).

## MANDATORY 5 confirmations (ALL must be true)
1. Structure aligned (1h + at least one of 4h/1d).
2. Entry in discount (LONG) / premium (SHORT) relative to the 1h range.
3. Entry sits INSIDE a same-direction OB or FVG (15m preferred; 1h/4h zone acceptable), ideally overlapping the POC zone.
4. Recent same-direction liquidity sweep (see swp above).
5. RSI supportive: LONG → 15m RSI < 65 (1h < 70); SHORT → 15m RSI > 35 (1h > 30). No opposite RSI divergence.
If any one fails → NO_SETUP for that coin.

## Entry
- entry_type = "CMP", entry = lc  ONLY when CMP is already inside the chosen zone (inPoc=true or lc between zone bounds).
- Otherwise entry_type = "LIMIT" at the best price inside the zone (POC/50% of the OB/FVG, fresh zone preferred):
  LONG limit must be BELOW CMP, SHORT limit must be ABOVE CMP, and ≤ 2.5% away. Farther → NO_SETUP.
- zone_high / zone_low = bounds of that 15m OB/FVG (for LIMIT).

## SL (15m structure — most important)
- LONG: SL below the 15m invalidation = lowest of {entry zone low, 15m sweep wick swp.cl, ...} minus 0.5×ATR(15m).
- SHORT: SL above the 15m invalidation = highest of {entry zone high, 15m sweep wick swp.ch, ...} plus 0.5×ATR(15m).
- Never use 1d/4h levels for SL. SL distance ≥ 0.8×ATR(15m) and ≤ 2% of entry. If a sane SL needs more than 2% → NO_SETUP.

## TP (day-trade, reachable today)
- TP1 = nearest liquidity pool (liqBS/eqh for LONG, liqSS/eql for SHORT) or nearest S/R cluster, whichever is closer (≤ 4% from entry).
- TP2 = next liquidity/S-R level or VAH/VAL. TP3 = opposite-side liquidity or range_high/range_low (≤ 10% from entry).
- LONG: sl < entry < tp1 < tp2 < tp3.  SHORT: sl > entry > tp1 > tp2 > tp3.
- rr_ratio = |tp1 − entry| / |entry − sl|   (TP1 based!). Must be ≥ 1.5, otherwise NO_SETUP. Do NOT quote RR of TP2/TP3.

## Confidence
- "high"   : 1d + 4h + 1h all aligned, sweep on 15m, fresh OB/FVG zone, RR ≥ 2.
- "medium" : 1h + one of 4h/1d aligned, everything else valid.
- Anything weaker → NO_SETUP (do not send "low").

## Output
- Pick at most ONE coin per group: the single best A+ setup. Call `send_trade_signal` with exact numbers and the
  coin's exact symbol as in the payload (e.g. "BTCUSDT").
- Before calling, self-check: direction vs premium/discount, sweep type vs direction, TP/SL ordering, RR to TP1 ≥ 1.5,
  and that your `reason` contains no contradictions. `reason` ≤ 350 characters, quote real payload values.
- If no coin qualifies, reply exactly "NO_SETUP"."""

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


def _norm_symbol(s: Any) -> str:
    s = str(s or "").upper().strip()
    s = s.replace("/", "").replace(":USDT", "").replace("-", "")
    return s


def _validate_signal(args: Dict[str, Any], symbols: List[str],
                     group_id: int) -> Optional[Dict[str, Any]]:
    """Make sure the model's tool call is complete & sane. Bad ones are dropped
    (otherwise they crash ranking / the trade monitor)."""
    try:
        sym = _norm_symbol(args.get("symbol"))
        if sym not in symbols:
            logger.warning("Group %d: signal symbol %r not in group %s — dropped",
                           group_id, args.get("symbol"), symbols)
            return None
        args["symbol"] = sym

        d = str(args.get("direction", "")).upper()
        if d not in ("LONG", "SHORT"):
            logger.warning("Group %d: bad direction %r — dropped", group_id, d)
            return None
        args["direction"] = d

        et = str(args.get("entry_type", "CMP")).upper()
        args["entry_type"] = et if et in ("CMP", "LIMIT") else "CMP"

        conf = str(args.get("confidence", "low")).lower()
        args["confidence"] = conf if conf in ("low", "medium", "high") else "low"

        for k in ("entry", "sl", "tp1", "tp2", "tp3"):
            args[k] = float(args[k])
        for k in ("zone_high", "zone_low", "poc_high", "poc_low", "sweep_level"):
            if args.get(k) is not None:
                try:
                    args[k] = float(args[k])
                except (TypeError, ValueError):
                    args[k] = None

        e, sl, t1, t2, t3 = (args["entry"], args["sl"], args["tp1"],
                             args["tp2"], args["tp3"])
        if min(e, sl, t1, t2, t3) <= 0:
            logger.warning("Group %d: %s non-positive price — dropped", group_id, sym)
            return None
        if d == "LONG" and not (sl < e < t1 < t2 < t3):
            logger.warning("Group %d: %s LONG levels invalid (sl=%s e=%s tp=%s/%s/%s) — dropped",
                           group_id, sym, sl, e, t1, t2, t3)
            return None
        if d == "SHORT" and not (sl > e > t1 > t2 > t3):
            logger.warning("Group %d: %s SHORT levels invalid (sl=%s e=%s tp=%s/%s/%s) — dropped",
                           group_id, sym, sl, e, t1, t2, t3)
            return None

        # always compute RR from the real numbers (model's own value can be wrong/missing)
        risk = abs(e - sl)
        args["rr_ratio"] = round(abs(t1 - e) / risk, 2) if risk > 0 else 0.0   # real RR to TP1, never the model's claim
        args["reason"] = str(args.get("reason", ""))
        return args
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Group %d: incomplete signal (%s) — dropped", group_id, exc)
        return None


def _retry_after(exc: Exception) -> Optional[float]:
    try:
        h = exc.response.headers.get("retry-after")  # type: ignore[attr-defined]
        return float(h) if h else None
    except Exception:
        return None


async def _analyze_group(client, key_idx: int, group_id: int,
                         symbols: List[str], payload: str,
                         data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Group {group_id} — analyse these {len(symbols)} coins:\n{payload}"},
    ]

    started = time.monotonic()
    effort = REASONING_EFFORT
    msg = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        wait = 0.0
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=OPENROUTER_MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    max_tokens=8000,
                    extra_body={"reasoning": {"effort": effort, "enabled": True}},
                ),
                timeout=_API_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("⌛ Group %d (key %d): timeout after %ds (attempt %d/%d)",
                           group_id, key_idx, _API_TIMEOUT, attempt, _MAX_ATTEMPTS)
            wait = 5
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status in (401, 402, 403):
                logger.error("❌ Group %d: API key %d rejected (%s). Skipping group.",
                             group_id, key_idx, status)
                return []
            if status == 429:
                wait = _retry_after(exc) or (20 * attempt)
                logger.warning("⏳ Group %d (key %d): rate limited 429 — retry in %.0fs (attempt %d/%d)",
                               group_id, key_idx, wait, attempt, _MAX_ATTEMPTS)
            else:
                wait = 5 * attempt + random.uniform(0, 3)
                logger.warning("Group %d (key %d): %s %s (attempt %d/%d)",
                               group_id, key_idx, status or type(exc).__name__,
                               str(exc)[:200], attempt, _MAX_ATTEMPTS)
        else:
            # OpenRouter can answer HTTP 200 with an {"error": ...} body and no choices
            choices = getattr(response, "choices", None)
            if not choices:
                err = getattr(response, "error", None)
                if err is None and getattr(response, "model_extra", None):
                    err = response.model_extra.get("error")
                logger.warning("Group %d (key %d): empty/error response body: %s (attempt %d/%d)",
                               group_id, key_idx, str(err)[:200], attempt, _MAX_ATTEMPTS)
                wait = 5 * attempt + random.uniform(0, 3)
            else:
                choice = choices[0]
                msg = choice.message
                finish = getattr(choice, "finish_reason", None)
                if not msg.tool_calls and not (msg.content or "").strip() \
                        and finish == "length":
                    # all 8000 tokens were eaten by reasoning → retry with less thinking
                    logger.warning("Group %d: output cut (finish=length, no answer) — retrying with lower reasoning",
                                   group_id)
                    effort = "medium" if effort == "high" else "low"
                    msg = None
                    wait = 2
                else:
                    break

        if attempt == _MAX_ATTEMPTS or (time.monotonic() - started) + wait > _GROUP_DEADLINE:
            break
        await asyncio.sleep(wait)

    if msg is None:
        logger.error("❌ Group %d (key %d) FAILED after retries — no result", group_id, key_idx)
        return []

    # OpenRouter sometimes returns reasoning in model_extra or directly
    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
    if not reasoning and getattr(msg, "model_extra", None):
        reasoning = msg.model_extra.get("reasoning") or msg.model_extra.get("reasoning_content")

    if reasoning:
        logger.info("🧠 Group %d reasoning (%d chars): %s …",
                    group_id, len(reasoning), reasoning[:300].replace("\n", " "))
        _save_reasoning(f"group{group_id}", reasoning)

    if not msg.tool_calls:
        content = (msg.content or "").strip()
        if content:
            logger.info("Group %d reply: %s", group_id, content[:150])
        else:
            logger.info("Group %d → NO_SETUP", group_id)
        return []

    signals: List[Dict[str, Any]] = []
    for tc in msg.tool_calls:
        if tc.function.name != "send_trade_signal":
            continue
        try:
            args = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Group %d: tool-call JSON could not be parsed", group_id)
            continue
        if not isinstance(args, dict):
            continue

        args = _validate_signal(args, symbols, group_id)
        if not args:
            continue

        if data and data.get(args["symbol"]):
            raw_sl = args["sl"]
            args, why = refine_signal(args, data[args["symbol"]])
            if not args:
                logger.info("🚫 Group %d: signal rejected by A+ filter — %s", group_id, why)
                continue
            logger.info("✅ Group %d: %s passed A+ filter (score %.1f) | model SL %s → 15m SL %s | RR(TP1) %.2f",
                        group_id, args["symbol"], args["score"], raw_sl, args["sl"], args["rr_ratio"])

        args["_group_id"] = group_id
        logger.info("🚀 Group %d signal: %s %s (%s, conf=%s, RR=%.2f)",
                    group_id, args["direction"], args["symbol"],
                    args["entry_type"], args["confidence"], args["rr_ratio"])
        signals.append(args)

    if not signals:
        logger.info("Group %d → no valid signal", group_id)
    return signals


def _rank_signals(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    conf_score = {"high": 3, "medium": 2, "low": 1}
    by_symbol: Dict[str, Dict] = {}
    for s in signals:
        sym = s["symbol"]
        if sym not in by_symbol:
            by_symbol[sym] = s
        else:
            if s.get("score", 0) > by_symbol[sym].get("score", 0):
                by_symbol[sym] = s

    return sorted(
        by_symbol.values(),
        key=lambda x: (float(x.get("score", 0)), conf_score.get(x.get("confidence"), 0),
                       float(x.get("rr_ratio", 0))),
        reverse=True,
    )


async def run_agent_batch(groups: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    clients = _get_clients()
    if not clients:
        return None

    n = min(len(clients), len(groups))

    async def _staggered_task(idx: int):
        # 1 request per second: group i is launched at i seconds.
        # After launch every group runs in parallel with its own API key
        # (no semaphore — that was serialising the calls before).
        await asyncio.sleep(idx * _LAUNCH_GAP_SEC)
        logger.info("📤 Launching group %d (key %d) at +%.0fs",
                    groups[idx]["id"], idx + 1, idx * _LAUNCH_GAP_SEC)
        return await _analyze_group(
            clients[idx], idx + 1, groups[idx]["id"],
            groups[idx]["symbols"], groups[idx]["payload"],
            groups[idx].get("data"),
        )

    tasks = [_staggered_task(i) for i in range(n)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_signals: List[Dict[str, Any]] = []
    failed = 0
    for r in results:
        if isinstance(r, Exception):
            failed += 1
            logger.error("Task crashed: %r", r, exc_info=r)
            continue
        if r:
            all_signals.extend(r)
    if failed:
        logger.warning("%d/%d groups crashed", failed, n)

    if not all_signals:
        logger.info("All groups → NO_SETUP (or failed)")
        return None

    ranked = _rank_signals(all_signals)
    best = ranked[0]

    logger.info("🏆 Best signal (highest score): %s %s score=%.1f conf=%s RR=%.2f from group %d",
                best["direction"], best["symbol"], best.get("score", 0), best.get("confidence"),
                best.get("rr_ratio", 0), best.get("_group_id", 0))

    if await send_signal(best):
        return best
    return None
