"""
V10 agent layer — Support/Resistance analysis by prompt (no script-side strategy filtering).

Stage 1: groups of 4 coins → one API call each (group i → key i, launches spaced 1/sec, bounded worker queue).
         The model returns AT MOST ONE best setup per group (or NO_SETUP).
Stage 2: if several groups returned a setup, a final "judge" call picks the single best. If the judge fails,
         the highest self-reported quality wins (ties → earliest group).
The script only checks FORMAT sanity (parsable JSON, known symbol, finite numbers, SL/TP on the right side).
It never rejects a setup because of RR, distance, trend, score or any other strategy criterion.
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

from config import OPENROUTER_API_KEYS, OPENROUTER_BASE_URL, OPENROUTER_MODEL, REASONING_EFFORT
from risk import check_levels, net_rr

logger = logging.getLogger(__name__)

_CLIENTS: List[AsyncOpenAI] = []
_SLEEP = asyncio.sleep                 # indirection so tests run without real delays
_API_TIMEOUT = 420                     # 4 coins × 550 candles + reasoning on a slow free model
_MAX_ATTEMPTS = 2
_GROUP_DEADLINE = 700
_LAUNCH_GAP_SEC = 1.0                  # one request START per second (group1 @0s, group2 @1s …)
_JUDGE_TIMEOUT = 240


def _get_clients() -> List[AsyncOpenAI]:
    global _CLIENTS
    if not _CLIENTS:
        _CLIENTS = [AsyncOpenAI(api_key=k, base_url=OPENROUTER_BASE_URL, max_retries=0,
                                default_headers={"HTTP-Referer": "https://localhost/sr-agent", "X-Title": "SR Agent v10"})
                    for k in OPENROUTER_API_KEYS]
    return _CLIENTS


# ───────────────────────────── prompts ─────────────────────────────
SYSTEM_PROMPT = """You are an expert price-action trader who trades ONLY horizontal Support/Resistance (S/R) zones.
No indicators, no ICT/SMC concepts — just raw OHLCV candles and volume. You receive up to 4 coins. For each coin you get
data_4h (200 candles), data_1h (200) and data_15m (150): {"T": candle OPEN time UTC "MM-DD,HH:MM", "O","H","L","C","V"},
oldest → newest. The LAST candle of each series is the latest CLOSED candle. Live prices are in the message header.
Use only this data. Never invent candles, levels or news.

## Method (follow in this order, for every coin)
1. 4h — map the MAJOR S/R ZONES (zones = price bands, not single lines): cluster swing highs/lows that reacted within roughly
   0.3–0.5 ATR of each other. Strength = number of clean reactions, size of the move away, volume on the reaction, recency,
   and role reversal (broken resistance → support and vice-versa). Read the trend: HH/HL, LH/LL or range.
2. 1h — refine those zones and add intermediate ones. Look for CONFLUENCE: a 1h zone inside/overlapping a 4h zone.
3. 15m — look for CONFIRMATION at a zone that price is at RIGHT NOW (inside it or within ~0.5 ATR(15m)):
   (a) rejection: long wick into the zone and a CLOSE back outside it / engulfing / strong 2-candle reversal, ideally on
       above-average volume; or (b) break-and-retest: a close through the zone, a retest of the flipped zone that holds.
   The confirming candle must already be CLOSED in the data. A mere touch is not confirmation.
4. Trade only HIGH-PROBABILITY zones. All of these should hold: zones agree on at least 2 timeframes (best: all 3); the zone
   has at least 2–3 clear historical reactions and is not worn out by many recent touches; direction is with the 4h/1h trend
   (counter-trend ONLY at a major 4h zone with a very strong confirmation); price is not mid-range/choppy; the path to TP1 is
   clean (no opposing zone between entry and TP1); you are not chasing (price has not already left the zone).
5. Entry: "CMP" if the confirmation just closed and live price is still in/near the zone; otherwise "LIMIT" at the zone's
   retest level — a LONG limit must be BELOW live price, a SHORT limit ABOVE it. Entry must sit inside/at the zone.
6. SL: beyond the FAR side of the zone (the level whose break invalidates the idea) plus a small buffer
   (~0.25–0.5 ATR(15m), or beyond the confirmation candle extreme if further). LONG: below support. SHORT: above resistance.
7. Targets = the NEXT S/R levels in the trade direction: TP1 = just before the nearest opposing zone, TP2 = the next zone,
   TP3 = the next major (4h) level / range extreme. Every target must be a real level visible in the data. Order: LONG
   sl < entry < tp1 < tp2 < tp3; SHORT sl > entry > tp1 > tp2 > tp3. Prefer setups with RR to TP1 of about 1.5 or better.

## Output
Compare ALL coins in this message and pick AT MOST ONE — the single best, highest-probability setup. If nothing is clearly
high-probability, reply exactly: NO_SETUP (that is a good answer; do not force trades).
Otherwise call `send_trade_signal` once with exact numbers. `confluence` = which timeframes' zones align (e.g. "4h+1h+15m
support 163.8-164.6"). `confirmation` = the confirming 15m candle(s) with their T time. `reason` ≤ 400 characters, factual,
citing real T times and prices from the data, no contradictions. `quality` = your own 0–100 rating of this setup versus an
ideal one (90+ only if everything is exceptional) — it is NOT a probability. Never promise or state a win probability."""

JUDGE_PROMPT = """You are the final selector. Below are trade setups produced by independent analyses of DIFFERENT coins; each is
already a Support/Resistance setup that passed a high-probability screen. Pick the SINGLE best one — the setup most likely to
reach TP1 and TP2 before its stop. Judge, in order: (1) multi-timeframe zone confluence and zone strength; (2) quality of the
15m confirmation; (3) alignment with the higher-timeframe trend; (4) clean path / room to TP1 and RR; (5) live price still
near the entry (not stale, not chased); (6) if several setups are the same market move (e.g. many alts long because BTC
bounces), prefer the cleanest one. Use only the information given. Reply by calling `select_best_signal`.
Never state a win probability."""

TOOLS = [{"type": "function", "function": {
    "name": "send_trade_signal",
    "description": "Report the single best Support/Resistance setup of this message.",
    "parameters": {"type": "object", "additionalProperties": False, "properties": {
        "symbol": {"type": "string", "description": "Exact coin symbol, e.g. SOLUSDT"},
        "direction": {"type": "string", "enum": ["LONG", "SHORT"]},
        "entry_type": {"type": "string", "enum": ["CMP", "LIMIT"]},
        "entry": {"type": "number"}, "zone_low": {"type": "number"}, "zone_high": {"type": "number"},
        "sl": {"type": "number"}, "tp1": {"type": "number"}, "tp2": {"type": "number"}, "tp3": {"type": "number"},
        "quality": {"type": "integer", "minimum": 0, "maximum": 100},
        "confluence": {"type": "string"}, "confirmation": {"type": "string"}, "reason": {"type": "string"},
    }, "required": ["symbol", "direction", "entry_type", "entry", "zone_low", "zone_high", "sl", "tp1", "tp2", "tp3",
                    "quality", "confluence", "confirmation", "reason"]}}}]

JUDGE_TOOLS = [{"type": "function", "function": {
    "name": "select_best_signal", "description": "Select the single best setup.",
    "parameters": {"type": "object", "additionalProperties": False, "properties": {
        "symbol": {"type": "string"}, "reason": {"type": "string"}}, "required": ["symbol", "reason"]}}}]


# ───────────────────────────── helpers ─────────────────────────────
def _save_reasoning(tag: str, reasoning: str):
    try:
        os.makedirs("debug_reasoning", exist_ok=True)
        with open(f"debug_reasoning/{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{tag}.txt", "w", encoding="utf-8") as f:
            f.write(reasoning)
    except Exception as exc:
        logger.warning("Could not save reasoning: %s", exc)


def _norm_symbol(x: Any) -> str:
    return str(x or "").upper().strip().replace("/", "").replace(":USDT", "").replace("-", "")


def sanity(args: Any, symbols: List[str], gid: Any) -> Tuple[Optional[Dict[str, Any]], str]:
    """FORMAT sanity only (parsable, known symbol, finite numbers, SL/TP on the correct side).
    Deliberately NO strategy criteria (RR, distance, trend, quality…) — the prompt decides what is a good setup."""
    if not isinstance(args, dict):
        return None, "tool arguments are not an object"
    sym = _norm_symbol(args.get("symbol"))
    if sym not in symbols:
        return None, f"symbol {args.get('symbol')!r} is not in group {gid} {symbols}"
    d = str(args.get("direction", "")).upper()
    try:
        n = {k: float(args[k]) for k in ("entry", "sl", "tp1", "tp2", "tp3")}
    except (KeyError, TypeError, ValueError):
        return None, f"{sym}: missing/non-numeric entry/sl/tp"
    err = check_levels(d, n["entry"], n["sl"], n["tp1"], n["tp2"], n["tp3"])
    if err:
        return None, f"{sym}: {err}"
    et = str(args.get("entry_type", "CMP")).upper()
    et = et if et in ("CMP", "LIMIT") else "CMP"
    zl, zh = args.get("zone_low"), args.get("zone_high")
    try:
        zl, zh = (min(float(zl), float(zh)), max(float(zl), float(zh)))
    except (TypeError, ValueError):
        zl = zh = None
    try:
        q = max(0, min(100, int(float(args.get("quality", 0)))))
    except (TypeError, ValueError):
        q = 0
    rr = net_rr(d, et, n["entry"], n["sl"], n["tp1"])
    return {"symbol": sym, "direction": d, "entry_type": et, **n, "zone_low": zl, "zone_high": zh, "quality": q,
            "score": q, "setup_type": "S/R", "confluence": str(args.get("confluence", ""))[:200],
            "confirmation": str(args.get("confirmation", ""))[:300], "reason": str(args.get("reason", ""))[:450],
            "net_rr": round(rr["net_rr"], 2), "gross_rr": round(rr["gross_rr"], 2), "risk_pct": round(rr["risk_pct"], 3),
            "rr_ratio": round(rr["net_rr"], 2)}, ""


def _retry_after(exc: Exception) -> Optional[float]:
    try:
        h = exc.response.headers.get("retry-after")  # type: ignore[attr-defined]
        return float(h) if h else None
    except Exception:
        return None


class _LaunchGate:
    """Global limiter: at most one request START per `gap` seconds across all workers (retries included)."""
    def __init__(self, gap: float):
        self.gap, self._next, self._lock = gap, 0.0, asyncio.Lock()

    async def wait(self):
        async with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.gap
        if delay:
            await _SLEEP(delay)


_GATE = _LaunchGate(_LAUNCH_GAP_SEC)


async def _chat(client, key_idx: int, tag: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]],
                gate: _LaunchGate, timeout: float, attempts: int, deadline: float) -> Tuple[Any, str]:
    """One model call with bounded retries (429/5xx/timeout/empty body/length). Returns (message | None, last_error)."""
    started, effort, err = time.monotonic(), REASONING_EFFORT, ""
    for attempt in range(1, attempts + 1):
        await gate.wait()
        wait = 0.0
        try:
            resp = await asyncio.wait_for(client.chat.completions.create(
                model=OPENROUTER_MODEL, messages=messages, tools=tools, tool_choice="auto", max_tokens=8000,
                extra_body={"reasoning": {"effort": effort, "enabled": True}}), timeout=timeout)
        except asyncio.TimeoutError:
            err, wait = f"timeout {timeout:.0f}s", 5
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status in (401, 402, 403):
                return None, f"key {key_idx} rejected ({status})"
            if status == 429:
                err, wait = "429 rate limited", _retry_after(exc) or 20 * attempt
            else:
                err, wait = f"{status or type(exc).__name__}: {str(exc)[:120]}", 5 * attempt
        else:
            choices = getattr(resp, "choices", None)
            if not choices:
                e = getattr(resp, "error", None) or (getattr(resp, "model_extra", None) or {}).get("error")
                err, wait = f"empty/error body: {str(e)[:120]}", 5 * attempt
            else:
                ch = choices[0]
                m = ch.message
                if not m.tool_calls and not (m.content or "").strip() and getattr(ch, "finish_reason", None) == "length":
                    effort = "medium" if effort == "high" else "low"
                    err, wait = "output cut by reasoning (length) → lower effort", 2
                else:
                    return m, ""
        logger.warning("%s (key %d) attempt %d/%d: %s", tag, key_idx, attempt, attempts, err)
        wait += random.uniform(0, 2) + 2 ** (attempt - 1)                # bounded exponential backoff + jitter
        if attempt == attempts or time.monotonic() - started + wait > deadline:
            break
        await _SLEEP(wait)
    return None, err


def _reasoning_of(msg: Any) -> Optional[str]:
    return getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) \
        or (getattr(msg, "model_extra", None) or {}).get("reasoning")


# ───────────────────────────── stage 1 ─────────────────────────────
async def analyze_group(client, key_idx: int, group: Dict[str, Any], gate: _LaunchGate = _GATE) -> Dict[str, Any]:
    """group = {'id', 'symbols', 'text'}. Returns {'group','key','status','candidates':[…],'error'}."""
    gid = group["id"]
    rep = {"group": gid, "key": key_idx, "status": "failed", "candidates": [], "error": ""}
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": group["text"]}]
    msg, err = await _chat(client, key_idx, f"Group {gid}", msgs, TOOLS, gate, _API_TIMEOUT, _MAX_ATTEMPTS, _GROUP_DEADLINE)
    if msg is None:
        rep["error"] = err
        logger.error("Group %d FAILED: %s", gid, err)
        return rep
    r = _reasoning_of(msg)
    if r:
        _save_reasoning(f"group{gid}", r)
    calls = [tc for tc in (msg.tool_calls or []) if tc.function.name == "send_trade_signal"]
    if not calls:
        rep["status"] = "no_setup"
        logger.info("Group %d → NO_SETUP", gid)
        return rep
    rep["status"] = "ok"
    for tc in calls:
        try:
            args = json.loads(tc.function.arguments)
        except Exception:
            rep["error"] = "tool-call JSON not parseable"
            logger.warning("Group %d: %s", gid, rep["error"])
            continue
        sig, why = sanity(args, group["symbols"], gid)
        if sig is None:
            rep["error"] = why
            logger.warning("Group %d: malformed signal dropped — %s", gid, why)
            continue
        sig["_group_id"] = gid
        rep["candidates"].append(sig)
        logger.info("Group %d → %s %s quality=%d netRR=%.2f", gid, sig["direction"], sig["symbol"], sig["quality"], sig["net_rr"])
    if not rep["candidates"]:
        rep["status"] = "invalid"
    return rep


async def run_agents(groups: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Bounded worker queue. Worker i uses key i, so group i → key i (1→15). More groups than keys → queued, never dropped."""
    clients = _get_clients()
    if not clients:
        logger.error("No OpenRouter keys configured")
        return {"candidates": [], "reports": [{"status": "failed", "error": "no API keys"}]}
    q: asyncio.Queue = asyncio.Queue()
    for g in groups:
        q.put_nowait(g)
    reports: List[Dict[str, Any]] = []

    async def worker(idx: int):
        while True:
            try:
                g = q.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                reports.append(await analyze_group(clients[idx], idx + 1, g))
            except Exception as exc:
                logger.error("Group %s crashed: %r", g.get("id"), exc, exc_info=True)
                reports.append({"group": g.get("id"), "key": idx + 1, "status": "failed", "candidates": [], "error": repr(exc)})

    await asyncio.gather(*[worker(i) for i in range(min(len(clients), len(groups)))])
    reports.sort(key=lambda r: r.get("group") or 0)
    cands = [c for r in reports for c in r.get("candidates", [])]
    cnt = lambda s: sum(1 for r in reports if r["status"] == s)
    logger.info("Groups: %d done · %d with setup · %d no_setup · %d failed · %d invalid",
                len(reports), cnt("ok"), cnt("no_setup"), cnt("failed"), cnt("invalid"))
    return {"candidates": cands, "reports": reports}


# ───────────────────────────── stage 2 ─────────────────────────────
def fallback_pick(cands: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Highest self-reported quality; ties → earliest group (stable)."""
    return sorted(cands, key=lambda c: (-c["quality"], c.get("_group_id", 0), c["symbol"]))[0]


def _judge_text(cands: List[Dict[str, Any]]) -> str:
    rows = [{"symbol": c["symbol"], "direction": c["direction"], "entry_type": c["entry_type"], "entry": c["entry"],
             "zone": [c["zone_low"], c["zone_high"]], "sl": c["sl"], "tp": [c["tp1"], c["tp2"], c["tp3"]],
             "net_rr_tp1": c["net_rr"], "quality": c["quality"], "confluence": c["confluence"],
             "confirmation": c["confirmation"], "reason": c["reason"]} for c in cands]
    return "Candidates:\n" + json.dumps(rows, separators=(",", ":"), ensure_ascii=False)


async def judge_best(cands: List[Dict[str, Any]], start_key: int = 0, tries: int = 3) -> Tuple[Optional[Dict[str, Any]], str]:
    """0 → None · 1 → itself (no extra call) · ≥2 → judge call (failover over keys) → fallback to best quality."""
    if not cands:
        return None, "no candidates"
    if len(cands) == 1:
        return cands[0], "single candidate"
    clients = _get_clients()
    by_sym = {c["symbol"]: c for c in cands}
    msgs = [{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": _judge_text(cands)}]
    for i in range(min(tries, len(clients))):
        ki = (start_key + i) % len(clients)
        msg, err = await _chat(clients[ki], ki + 1, "Judge", msgs, JUDGE_TOOLS, _GATE, _JUDGE_TIMEOUT, 1, _JUDGE_TIMEOUT)
        if msg is None:
            logger.warning("Judge key %d failed: %s", ki + 1, err)
            continue
        for tc in msg.tool_calls or []:
            try:
                sym = _norm_symbol(json.loads(tc.function.arguments).get("symbol"))
            except Exception:
                continue
            if sym in by_sym:
                logger.info("⚖️ Judge picked %s", sym)
                return by_sym[sym], "judge"
        logger.warning("Judge key %d returned no valid pick", ki + 1)
    best = fallback_pick(cands)
    logger.warning("Judge unavailable → highest quality: %s", best["symbol"])
    return best, "fallback:max-quality"
