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
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from config import (
    OPENROUTER_API_KEYS, OPENROUTER_BASE_URL, OPENROUTER_MODEL,
    REASONING_EFFORT, ENGINE_ONLY_MODE, TOOL_STRICT_SCHEMA,
)
from risk import check_levels

logger = logging.getLogger(__name__)

_CLIENTS: List[AsyncOpenAI] = []
_SLEEP = asyncio.sleep       # indirection so tests can run without real delays
_API_TIMEOUT = 240          # Nemotron Ultra is slow (reasoning) — 180s was killing many calls
_MAX_ATTEMPTS = 2           # per group (same key)
_GROUP_DEADLINE = 420       # stop retrying a group after this many seconds
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


SYSTEM_PROMPT = """You are a strict reviewer of SCALP setups (ICT/SMC, crypto futures). A deterministic Python engine has ALREADY
confirmed, on CLOSED candles only: 1H major trend/bias → 15m liquidity sweep+reclaim → 15m displacement MSS → fresh OB/FVG zone
→ 5m entry confirmation (5m MSS + 5m FVG inside the 15m zone) → 15m-structure stop → structural targets.
All numbers are computed by code. Your job is judgment, not arithmetic. Only the best-quality setup should survive.

## Input (compact JSON, one entry per confirmed setup)
s symbol | dir | type (continuation = with 1H trend) | et entry type (CMP=market, LIMIT) | e entry | z [zone low, high]
sl stop | tp [tp1,tp2,tp3] and tps (what liquidity each target is) | nrr net RR to TP1 after fees/slippage/funding | risk% SL distance
ctx {b1 = 1H bias, b15 = 15m bias (+1 bull/-1 bear/0), r1 = 1H dealing range [lo, eq, hi]} | atr ATR(15m) | px live price
sw {level, src, ts, wick} = swept liquidity pool | mss {ref, ts, disp = displacement body in ATR, rv = relative volume, age}
zone {kind OB/FVG/OB+FVG, touches} | r5 5m confirmation (confirmed) | c15 / c5 = last closed 15m / 5m candles [o,h,l,c], oldest first

## How to choose (be harsh — win rate matters more than frequency)
Prefer: obvious higher-timeframe liquidity swept (src 1h:*), strong displacement (disp ≥ 1.2, rv ≥ 1.2), fresh zone (touches 0),
price already close to the entry, 1H and 15m bias both with the trade, targets on real liquidity with room before opposing levels,
clean candles in c15/c5 (no big opposing wick/absorption at the entry). Reject: choppy/overlapping candles, an entry that already
looks tested several times, a stop that is inside noise, targets blocked by nearby opposing liquidity, anything contradictory.
If no setup is clearly A-grade → reply exactly: NO_SETUP. Otherwise pick AT MOST ONE: the single best.

## Output
Call `send_trade_signal` once, COPYING the chosen setup's numbers exactly (symbol, direction, entry_type, entry, zone_high,
zone_low, sl, tp1, tp2, tp3). Never alter or invent levels — the server rejects any mismatch. `reason` ≤ 300 chars, factual,
only values present in the input. `confidence`/`rr_ratio` are informational; the server ignores and recomputes them.
Never state a win probability."""

TOOLS = [{
    "type": "function",
    "function": {
        **({"strict": True} if TOOL_STRICT_SCHEMA else {}),
        "name": "send_trade_signal",
        "description": "Select ONE engine-confirmed scalp setup (numbers must be copied exactly).",
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
            "additionalProperties": False,
            "required": [
                "symbol", "direction", "entry_type", "entry",
                "sl", "tp1", "tp2", "tp3", "reason",
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


def _norm_symbol(x: Any) -> str:
    return str(x or "").upper().strip().replace("/", "").replace(":USDT", "").replace("-", "")


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def validate_against_candidate(args: Dict[str, Any], cands: Dict[str, Dict[str, Any]],
                               group_id: int) -> Tuple[Optional[Dict[str, Any]], str]:
    """Server-side contract check. The model may only SELECT an engine candidate; numbers must match it.
    Returns (final_signal, '') or (None, exact_rejection_reason). Model rr/confidence are never trusted."""
    if not isinstance(args, dict):
        return None, "tool arguments are not an object"
    sym = _norm_symbol(args.get("symbol"))
    cand = cands.get(sym)
    if cand is None:
        return None, f"symbol {args.get('symbol')!r} is not a candidate of group {group_id}"
    req = ("direction", "entry_type", "entry", "sl", "tp1", "tp2", "tp3", "reason")
    miss = [k for k in req if args.get(k) in (None, "")]
    if miss:
        return None, f"{sym}: missing required field(s) {miss}"
    try:
        nums = {k: float(args[k]) for k in ("entry", "sl", "tp1", "tp2", "tp3")}
    except (TypeError, ValueError):
        return None, f"{sym}: non-numeric price field"
    if str(args["direction"]).upper() != cand["direction"]:
        return None, f"{sym}: direction {args['direction']} contradicts engine {cand['direction']}"
    if str(args["entry_type"]).upper() != cand["entry_type"]:
        return None, f"{sym}: entry_type {args['entry_type']} contradicts engine {cand['entry_type']}"
    err = check_levels(cand["direction"], nums["entry"], nums["sl"], nums["tp1"], nums["tp2"], nums["tp3"])
    if err:
        return None, f"{sym}: {err}"
    tol = 0.1 * cand["atr15"]
    for k in ("entry", "sl", "tp1", "tp2", "tp3"):
        if not _close(nums[k], cand[k], tol):
            return None, f"{sym}: {k}={nums[k]} does not match engine value {cand[k]} (tol {tol:.6g})"
    out = {k: v for k, v in cand.items() if not k.startswith("_")}
    out["model_confidence"] = str(args.get("confidence", ""))
    out["reason"] = str(args["reason"])[:350]
    out["confidence"] = "high" if cand["score"] >= 80 else "medium" if cand["score"] >= 70 else "standard"
    out["rr_ratio"] = cand["net_rr"]
    out["_group_id"] = group_id
    return out, ""


def _retry_after(exc: Exception) -> Optional[float]:
    try:
        h = exc.response.headers.get("retry-after")  # type: ignore[attr-defined]
        return float(h) if h else None
    except Exception:
        return None


class _LaunchGate:
    """Global rate limiter: at most one request START per `gap` seconds across ALL workers (incl. retries)."""
    def __init__(self, gap: float):
        self.gap, self._next, self._lock = gap, 0.0, asyncio.Lock()

    async def wait(self):
        async with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.gap
        if delay:
            await _SLEEP(delay)


async def _analyze_group(client, key_idx: int, group: Dict[str, Any], gate: _LaunchGate) -> Dict[str, Any]:
    gid, cands = group["id"], group["cands"]
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Group {gid} candidates:\n{group['payload']}"}]
    started, effort, msg = time.monotonic(), REASONING_EFFORT, None
    report = {"group": gid, "key": key_idx, "status": "failed", "error": "", "signal": None}

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        await gate.wait()
        wait, err = 0.0, ""
        try:
            tool = TOOLS
            response = await asyncio.wait_for(client.chat.completions.create(
                model=OPENROUTER_MODEL, messages=messages, tools=tool, tool_choice="auto", max_tokens=8000,
                extra_body={"reasoning": {"effort": effort, "enabled": True}}), timeout=_API_TIMEOUT)
        except asyncio.TimeoutError:
            err, wait = f"timeout {_API_TIMEOUT}s", 5
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status in (401, 402, 403):
                report["error"] = f"key {key_idx} rejected ({status})"
                logger.error("Group %d: %s", gid, report["error"])
                return report
            if status == 429:
                wait = _retry_after(exc) or 20 * attempt
                err = "429 rate limited"
            else:
                err, wait = f"{status or type(exc).__name__}: {str(exc)[:120]}", 5 * attempt
        else:
            choices = getattr(response, "choices", None)
            if not choices:
                e = getattr(response, "error", None) or (getattr(response, "model_extra", None) or {}).get("error")
                err, wait = f"empty/error body: {str(e)[:120]}", 5 * attempt
            else:
                ch = choices[0]
                m = ch.message
                if not m.tool_calls and not (m.content or "").strip() and getattr(ch, "finish_reason", None) == "length":
                    effort = "medium" if effort == "high" else "low"
                    err, wait = "output cut by reasoning (length) → lower effort", 2
                else:
                    msg = m
                    break
        report["error"] = err
        logger.warning("Group %d (key %d) attempt %d/%d: %s", gid, key_idx, attempt, _MAX_ATTEMPTS, err)
        wait += random.uniform(0, 2) + 2 ** (attempt - 1)               # bounded exponential backoff + jitter
        if attempt == _MAX_ATTEMPTS or time.monotonic() - started + wait > _GROUP_DEADLINE:
            break
        await _SLEEP(wait)

    if msg is None:
        logger.error("Group %d FAILED after retries: %s", gid, report["error"])
        return report

    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) \
        or (getattr(msg, "model_extra", None) or {}).get("reasoning")
    if reasoning:
        _save_reasoning(f"group{gid}", reasoning)

    report["status"], report["error"] = "ok", ""
    if not msg.tool_calls:
        logger.info("Group %d → NO_SETUP (%s)", gid, (msg.content or "").strip()[:80])
        report["status"] = "no_setup"
        return report
    tcs = [tc for tc in msg.tool_calls if tc.function.name == "send_trade_signal"]
    if len(tcs) > 1:
        logger.warning("Group %d: model returned %d signals, only the first is considered", gid, len(tcs))
    try:
        args = json.loads(tcs[0].function.arguments)
    except Exception:
        report["status"], report["error"] = "invalid", "tool-call JSON not parseable"
        return report
    sig, why = validate_against_candidate(args, cands, gid)
    if sig is None:
        logger.warning("🚫 Group %d signal rejected: %s", gid, why)
        report["status"], report["error"] = "rejected", why
        return report
    logger.info("✅ Group %d accepted: %s %s score=%.1f netRR=%.2f", gid, sig["direction"], sig["symbol"],
                sig["score"], sig["net_rr"])
    report["signal"] = sig
    return report


def rank_signals(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deterministic: heuristic quality score → net RR → symbol (stable). Not a win probability."""
    return sorted(signals, key=lambda x: (-float(x["score"]), -float(x["net_rr"]), x["symbol"]))


async def run_agent_batch(groups: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Bounded worker queue: every group is processed even if groups > keys. Returns {'best', 'reports'}."""
    if ENGINE_ONLY_MODE:
        sigs = []
        for g in groups:
            c = max(g["cands"].values(), key=lambda x: x["score"])
            sigs.append({**{k: v for k, v in c.items() if not k.startswith("_")}, "reason": "engine-only (no LLM review)",
                         "confidence": "standard", "rr_ratio": c["net_rr"], "_group_id": g["id"]})
        return {"best": rank_signals(sigs)[0] if sigs else None, "reports": []}

    clients = _get_clients()
    if not clients:
        logger.error("No OpenRouter keys — cannot review candidates (fail closed)")
        return {"best": None, "reports": [{"status": "failed", "error": "no API keys"}]}
    q: asyncio.Queue = asyncio.Queue()
    for g in groups:
        q.put_nowait(g)
    gate = _LaunchGate(_LAUNCH_GAP_SEC)
    reports: List[Dict[str, Any]] = []

    async def worker(idx: int):
        while True:
            try:
                g = q.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                reports.append(await _analyze_group(clients[idx], idx + 1, g, gate))
            except Exception as exc:
                logger.error("Group %s crashed: %r", g.get("id"), exc, exc_info=True)
                reports.append({"group": g.get("id"), "status": "failed", "error": repr(exc), "signal": None})

    await asyncio.gather(*[worker(i) for i in range(min(len(clients), len(groups)))])
    failed = [r for r in reports if r["status"] in ("failed", "invalid")]
    logger.info("Groups: %d processed, %d failed, %d rejected, %d no_setup",
                len(reports), len(failed), sum(r["status"] == "rejected" for r in reports),
                sum(r["status"] == "no_setup" for r in reports))
    sigs = [r["signal"] for r in reports if r.get("signal")]
    return {"best": rank_signals(sigs)[0] if sigs else None, "reports": reports}


_GATE = _LaunchGate(_LAUNCH_GAP_SEC)


async def run_group(group: Dict[str, Any], key_index: int) -> Dict[str, Any]:
    """Review ONE payload with ONE specific api key (called per scan by main's round-robin)."""
    clients = _get_clients()
    if ENGINE_ONLY_MODE:
        c = max(group["cands"].values(), key=lambda x: x["score"])
        sig = {**{k: v for k, v in c.items() if not k.startswith("_")}, "reason": "engine-only (no LLM review)",
               "confidence": "standard", "rr_ratio": c["net_rr"], "_group_id": group["id"]}
        return {"status": "ok", "signal": sig, "error": ""}
    if not clients:
        return {"status": "failed", "signal": None, "error": "no API keys"}
    return await _analyze_group(clients[key_index % len(clients)], key_index % len(clients) + 1, group, _GATE)
