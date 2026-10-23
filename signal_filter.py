"""
A+ gatekeeper. The LLM *proposes* a trade; this module VERIFIES it against the
computed indicator data, places the SL on 15m structure, recomputes RR from the
real numbers and gives a 0-100 quality score. Weak / inconsistent setups are dropped.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

from config import (
    MIN_RR, MAX_SL_PCT, SL_ATR_BUFFER, SL_MIN_ATR, MAX_ENTRY_DIST_PCT,
    MAX_TP1_PCT, MAX_TP3_PCT, MIN_SCORE,
)

logger = logging.getLogger(__name__)


def _f(v) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _bias(t: Dict[str, Any]) -> int:
    """+1 bull / -1 bear / 0 unknown — latest structure event, else EMA trend."""
    ls = t.get("last_struct")
    if isinstance(ls, dict) and ls.get("d") in (1, -1):
        return int(ls["d"])
    return {"bull": 1, "bear": -1}.get(t.get("ema_trend"), 0)


def _zones(t: Dict[str, Any], want: str) -> List[Tuple[str, float, float]]:
    out = []
    for key, kind in (("ob_list", "OB"), ("fvg_list", "FVG")):
        for z in t.get(key) or []:
            lo, hi = _f(z.get("bot")), _f(z.get("top"))
            if z.get("t") == want and lo is not None and hi is not None:
                out.append((kind, min(lo, hi), max(lo, hi)))
    return out


def refine_signal(sig: Dict[str, Any], sym_data: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """Returns (refined_signal | None, reason_if_rejected)."""
    sym, d = sig["symbol"], sig["direction"]
    dr = 1 if d == "LONG" else -1
    want = "bull" if dr == 1 else "bear"
    t15, t1h, t4h, t1d = (sym_data.get(k) or {} for k in ("15m", "1h", "4h", "1d"))

    entry = sig["entry"]
    cmp_ = _f(t15.get("last_close"))
    atr15 = _f(t15.get("atr"))
    if not cmp_ or not atr15:
        return None, "no 15m price/ATR data"

    # ── entry sanity ────────────────────────────────────────────
    dist = abs(entry - cmp_) / cmp_ * 100
    if sig["entry_type"] == "LIMIT":
        if dr == 1 and entry > cmp_ * 1.0005:
            return None, f"LONG limit {entry} is ABOVE price {cmp_}"
        if dr == -1 and entry < cmp_ * 0.9995:
            return None, f"SHORT limit {entry} is BELOW price {cmp_}"
        if dist > MAX_ENTRY_DIST_PCT:
            return None, f"limit entry {dist:.2f}% away (> {MAX_ENTRY_DIST_PCT}%)"
    else:
        if dist > 0.3:
            return None, f"CMP entry {entry} is {dist:.2f}% from price {cmp_}"

    # ── confirmation 2: premium / discount (entry price vs 1H equilibrium) ──
    eq1h = _f(t1h.get("equilibrium"))
    if eq1h is not None:
        if dr == -1 and entry < eq1h:
            return None, f"SHORT entry {entry} is in DISCOUNT (1H eq {eq1h})"
        if dr == 1 and entry > eq1h:
            return None, f"LONG entry {entry} is in PREMIUM (1H eq {eq1h})"

    # ── confirmation 1: HTF structure alignment ─────────────────
    b1h, b4h, b1d = _bias(t1h), _bias(t4h), _bias(t1d)
    aligned = [b == dr for b in (b1h, b4h, b1d)]
    if b1h == -dr:
        return None, "1H structure is against the trade"
    if sum(aligned) < 2:
        return None, f"HTF alignment too weak (1H={b1h} 4H={b4h} 1D={b1d})"

    # ── confirmation 3: OB / FVG zone containing the entry ──────
    tol = 0.25 * atr15
    m15 = [z for z in _zones(t15, want) if z[1] - tol <= entry <= z[2] + tol]
    mh = [z for tf in (t1h, t4h) for z in _zones(tf, want) if z[1] - tol <= entry <= z[2] + tol]
    if not m15 and not mh:
        return None, f"entry {entry} not inside any {want} OB/FVG"
    kinds = {z[0] for z in (m15 or mh)}

    # ── confirmation 4: liquidity sweep in the trade direction ──
    s15, s1h = t15.get("recent_sweep") or {}, t1h.get("recent_sweep") or {}
    sweep15 = s15.get("t") == want
    sweep1h = s1h.get("t") == want
    if not (sweep15 or sweep1h):
        return None, f"no {want} sweep (15m={s15.get('t')} 1H={s1h.get('t')})"

    # ── confirmation 5: RSI ─────────────────────────────────────
    r15, r1h = _f(t15.get("rsi")), _f(t1h.get("rsi"))
    if r15 is not None and ((dr == 1 and r15 >= 70) or (dr == -1 and r15 <= 30)):
        return None, f"15m RSI {r15} extreme against {d}"
    if r1h is not None and ((dr == 1 and r1h >= 75) or (dr == -1 and r1h <= 25)):
        return None, f"1H RSI {r1h} extreme against {d}"

    # ── SL on 15m structure ─────────────────────────────────────
    anchors: List[float] = []
    for kind, lo, hi in m15:
        anchors.append(lo if dr == 1 else hi)
    zl, zh = _f(sig.get("zone_low")), _f(sig.get("zone_high"))
    if sig["entry_type"] == "LIMIT" and zl is not None and zh is not None \
            and zl - tol <= entry <= zh + tol:
        anchors.append(zl if dr == 1 else zh)
    if sweep15:
        w = _f(s15.get("cl" if dr == 1 else "ch"))
        if w is not None and ((dr == 1 and w < entry) or (dr == -1 and w > entry)):
            anchors.append(w)
    if not anchors:
        sw = _f(t15.get("swing_low" if dr == 1 else "swing_high"))
        if sw is not None and ((dr == 1 and sw < entry) or (dr == -1 and sw > entry)):
            anchors.append(sw)
    if not anchors:
        return None, "no 15m structure to anchor SL"

    buf = SL_ATR_BUFFER * atr15
    if dr == 1:
        sl = min(min(anchors) - buf, entry - SL_MIN_ATR * atr15)
    else:
        sl = max(max(anchors) + buf, entry + SL_MIN_ATR * atr15)

    risk = abs(entry - sl)
    risk_pct = risk / entry * 100
    if risk_pct > MAX_SL_PCT:
        return None, f"SL too wide: {risk_pct:.2f}% (> {MAX_SL_PCT}%)"

    # ── TPs / RR from the real numbers ──────────────────────────
    tp1, tp3 = sig["tp1"], sig["tp3"]
    tp1_pct, tp3_pct = abs(tp1 - entry) / entry * 100, abs(tp3 - entry) / entry * 100
    if tp1_pct > MAX_TP1_PCT or tp3_pct > MAX_TP3_PCT:
        return None, f"TPs not day-trade reachable (TP1 {tp1_pct:.1f}%, TP3 {tp3_pct:.1f}%)"
    rr = abs(tp1 - entry) / risk
    if rr < MIN_RR:
        return None, f"RR to TP1 only {rr:.2f} (< {MIN_RR})"

    # ── score (0-100) ───────────────────────────────────────────
    sc = 0.0
    sc += 12 * (b1h == dr) + 10 * (b4h == dr) + 8 * (b1d == dr)           # HTF   30
    sc += 10 + (5 if (_f(t4h.get("equilibrium")) is not None and
                      ((dr == 1 and entry <= _f(t4h["equilibrium"])) or
                       (dr == -1 and entry >= _f(t4h["equilibrium"])))) else 0)  # P/D   15
    sc += min(15, (9 if "OB" in kinds else 0) + (8 if "FVG" in kinds else 0))     # zone  15
    sc += 15 if sweep15 else 9                                              # sweep 15
    ok15 = r15 is None or (r15 < 65 if dr == 1 else r15 > 35)
    ok1h = r1h is None or (r1h < 70 if dr == 1 else r1h > 30)
    sc += 5 * ok15 + 5 * ok1h                                               # RSI   10
    sc += min(10, 5 + (rr - MIN_RR) / 1.5 * 5)                              # RR    10
    poc_lo, poc_hi = _f(t15.get("poc_zone_low")), _f(t15.get("poc_zone_high"))
    if poc_lo is not None and poc_hi is not None and poc_lo <= entry <= poc_hi:
        sc += 3                                                             # entry 5
    if (_f(t15.get("rv")) or 0) >= 1.0:
        sc += 2
    sc += {"high": 5, "medium": 3, "low": 1}.get(sig.get("confidence"), 0)  # conf   5
    sc = round(min(100.0, sc), 1)

    if sc < MIN_SCORE:
        return None, f"score {sc} < {MIN_SCORE}"

    out = dict(sig)
    out["model_sl"] = sig["sl"]
    out["sl"] = round(sl, 8)
    out["rr_ratio"] = round(rr, 2)
    out["score"] = sc
    out["sl_basis"] = "15m structure"
    return out, ""
