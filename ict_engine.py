"""
Deterministic, CAUSAL ICT/SMC engine.

Definitions (all computed on CLOSED candles only; nothing uses data after the evaluation time)
-------------------------------------------------------------------------------------------
swing        pivot high at i: high[i] > every high in the `left` candles before it and >= every high in the
             `right` candles after it. It is CONFIRMED (usable) only at candle i+right. Same for lows.
equal pool   >=2 confirmed swing highs (lows) within EQ_TOL_ATR*ATR of each other, not yet closed through.
bias         HH+HL => +1, LH+LL => -1 (last two confirmed swings each); else last close beyond last swing.
dealing rng  last confirmed swing high & last confirmed swing low of the timeframe; eq = midpoint.
sweep        a closed candle whose wick pierces a pre-existing, never-touched liquidity level by
             [SWEEP_MIN_PEN_ATR, SWEEP_MAX_PEN_ATR]*ATR and which (or one of the next SWEEP_RECLAIM_BARS
             candles) CLOSES back on the original side. A wick alone / a close-through is not a sweep.
displacement a candle with body >= DISP_BODY_ATR*ATR and body >= DISP_BODY_RANGE of its range, in trade direction.
MSS          first CLOSE beyond the nearest confirmed opposite swing that preceded the sweep, with a
             displacement candle inside the leg (sweep->break). One component; BOS/CHoCH are not scored again.
zone         FVG (3-candle gap) and/or OB (last opposite candle before the displacement) formed in the leg.
"""
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config as C
from ohlcv import TF_MS
from risk import net_rr, blended_net_rr, check_levels

Candle = pd.DataFrame


# ─────────────────────────── primitives ───────────────────────────
def atr_series(df: Candle, n: int = C.ATR_LEN) -> np.ndarray:
    h, l, c = (df[x].to_numpy(float) for x in ("high", "low", "close"))
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    out = np.full(len(df), np.nan)
    if len(df) > n:
        cs = np.cumsum(tr)
        out[n:] = (cs[n:] - cs[:-n]) / n
    return out


def find_swings(df: Candle, left: int, right: int) -> List[Dict[str, Any]]:
    h, l, t = df["high"].to_numpy(float), df["low"].to_numpy(float), df["t"].to_numpy(np.int64)
    out: List[Dict[str, Any]] = []
    for i in range(left, len(df) - right):
        if h[i] > h[i - left:i].max() and h[i] >= h[i + 1:i + right + 1].max():
            out.append({"i": i, "t": int(t[i]), "price": float(h[i]), "kind": "H", "conf_i": i + right})
        if l[i] < l[i - left:i].min() and l[i] <= l[i + 1:i + right + 1].min():
            out.append({"i": i, "t": int(t[i]), "price": float(l[i]), "kind": "L", "conf_i": i + right})
    return out


def structure_bias(df: Candle, sw: List[Dict[str, Any]]) -> int:
    hs = [s for s in sw if s["kind"] == "H"][-2:]
    ls = [s for s in sw if s["kind"] == "L"][-2:]
    if len(hs) == 2 and len(ls) == 2:
        if hs[1]["price"] > hs[0]["price"] and ls[1]["price"] > ls[0]["price"]:
            return 1
        if hs[1]["price"] < hs[0]["price"] and ls[1]["price"] < ls[0]["price"]:
            return -1
    c = float(df["close"].iloc[-1])
    if hs and c > hs[-1]["price"]:
        return 1
    if ls and c < ls[-1]["price"]:
        return -1
    return 0


def dealing_range(sw: List[Dict[str, Any]], atr: float) -> Optional[Dict[str, float]]:
    hs = [s for s in sw if s["kind"] == "H"]
    ls = [s for s in sw if s["kind"] == "L"]
    if not hs or not ls or not atr or math.isnan(atr):
        return None
    hi, lo = hs[-1]["price"], ls[-1]["price"]
    if hi <= lo or (hi - lo) < 1.5 * atr:
        return None
    return {"hi": hi, "lo": lo, "eq": (hi + lo) / 2.0}


def equal_pools(df: Candle, sw: List[Dict[str, Any]], kind: str, atr: float) -> List[Dict[str, Any]]:
    """Equal highs ('H') / lows ('L') that have not been closed through since the last touch."""
    if not atr or math.isnan(atr):
        return []
    pts = [s for s in sw if s["kind"] == kind][-12:]
    tol = C.EQ_TOL_ATR * atr
    pts = sorted(pts, key=lambda s: s["price"])
    groups, cur = [], []
    for p in pts:
        if cur and p["price"] - cur[-1]["price"] <= tol:
            cur.append(p)
        else:
            if len(cur) >= 2:
                groups.append(cur)
            cur = [p]
    if len(cur) >= 2:
        groups.append(cur)
    close = df["close"].to_numpy(float)
    out = []
    for g in groups:
        lvl = max(x["price"] for x in g) if kind == "H" else min(x["price"] for x in g)
        last_i = max(x["i"] for x in g)
        after = close[last_i + 1:]
        if len(after) and ((kind == "H" and after.max() > lvl) or (kind == "L" and after.min() < lvl)):
            continue
        out.append({"price": lvl, "n": len(g), "last_i": last_i,
                    "avail_t": int(df["t"].iloc[min(last_i, len(df) - 1)])})
    return out


def _iso(ms) -> str:
    if isinstance(ms, str):
        return ms
    return pd.to_datetime(int(ms), unit="ms", utc=True).strftime("%Y-%m-%d %H:%MZ")


def _py(o):
    """numpy → plain python (JSON-safe), recursively."""
    if isinstance(o, dict):
        return {k: _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


# ─────────────────────── liquidity levels (per TF) ───────────────────────
def tf_levels(df: Candle, tf: str, side: str, price: float) -> List[Dict[str, Any]]:
    """Liquidity levels. side='L' → sell-side (lows, below price); 'H' → buy-side (highs, above)."""
    step, r = TF_MS[tf], C.SWING_LR[tf]
    atr = atr_series(df)[-1]
    sw = find_swings(df, r, r)
    out = []
    for s in [x for x in sw if x["kind"] == side][-12:]:
        out.append({"price": s["price"], "src": f"{tf}:swing", "tf": tf, "t": s["t"],
                    "avail_t": s["t"] + (r + 1) * step})
    for p in equal_pools(df, sw, side, atr):
        out.append({"price": p["price"], "src": f"{tf}:eq{'h' if side == 'H' else 'l'}x{p['n']}", "tf": tf,
                    "t": p["avail_t"], "avail_t": p["avail_t"] + step})
    lim = C.MAX_TARGET_PCT / 100.0
    return [x for x in out if abs(x["price"] - price) / price <= lim]


# ─────────────────────────── sweep / MSS / zones ───────────────────────────
def find_sweep(df15: Candle, atr: np.ndarray, direction: int, levels: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """direction +1 (LONG): sweep of sell-side level (low pierced, close reclaimed). -1 mirrored."""
    t = df15["t"].to_numpy(np.int64)
    hi, lo, cl = (df15[x].to_numpy(float) for x in ("high", "low", "close"))
    n = len(df15)
    best = None
    sig = {"4h": 3, "1h": 2, "15m": 1}
    for k in range(n - 1, max(n - 1 - C.SWEEP_LOOKBACK_BARS, 1), -1):
        a = atr[k]
        if not a or math.isnan(a):
            continue
        for lv in levels:
            p = lv["price"]
            if t[k] < lv["avail_t"]:
                continue                                    # level not yet known at candle k
            first = int(np.searchsorted(t, lv["avail_t"]))
            seg_lo, seg_hi, seg_cl = lo[first:k], hi[first:k], cl[first:k]
            if direction == 1:
                pen = p - lo[k]
                if not (C.SWEEP_MIN_PEN_ATR * a <= pen <= C.SWEEP_MAX_PEN_ATR * a):
                    continue
                if len(seg_lo) and seg_lo.min() < p:        # level already touched/consumed earlier
                    continue
                rec = next((r for r in range(k, min(k + C.SWEEP_RECLAIM_BARS, n - 1) + 1) if cl[r] > p), None)
                if rec is None:
                    continue
                wick = float(lo[k:rec + 1].min())
            else:
                pen = hi[k] - p
                if not (C.SWEEP_MIN_PEN_ATR * a <= pen <= C.SWEEP_MAX_PEN_ATR * a):
                    continue
                if len(seg_hi) and seg_hi.max() > p:
                    continue
                rec = next((r for r in range(k, min(k + C.SWEEP_RECLAIM_BARS, n - 1) + 1) if cl[r] < p), None)
                if rec is None:
                    continue
                wick = float(hi[k:rec + 1].max())
            cand = {"k": k, "rec": rec, "level": p, "src": lv["src"], "tf": lv["tf"], "wick": wick,
                    "pen_atr": pen / a, "sweep_ts": int(t[k]), "reclaim_ts": int(t[rec]),
                    "level_ts": int(lv["t"]), "sig": sig.get(lv["tf"], 1)}
            key = (cand["k"], cand["sig"])
            if best is None or key > (best["k"], best["sig"]):
                best = cand
        if best is not None and best["k"] == k:
            break                                           # most recent candle with a sweep wins
    return best


def find_mss(df15: Candle, atr: np.ndarray, sw15: List[Dict[str, Any]], sweep: Dict[str, Any],
             direction: int) -> Optional[Dict[str, Any]]:
    k, n = sweep["k"], len(df15)
    t = df15["t"].to_numpy(np.int64)
    o, hi, lo, cl = (df15[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    want = "H" if direction == 1 else "L"
    refs = [s for s in sw15 if s["kind"] == want and s["i"] < k]
    if not refs:
        return None
    ref = refs[-1]
    j = next((x for x in range(k, n) if (cl[x] > ref["price"] if direction == 1 else cl[x] < ref["price"])), None)
    if j is None:
        return None
    # sweep must still be valid: nothing closed through the wick extreme afterwards
    after = cl[sweep["rec"] + 1:]
    if len(after) and ((direction == 1 and after.min() < sweep["wick"]) or
                       (direction == -1 and after.max() > sweep["wick"])):
        return None
    best_d, best_body = None, 0.0
    for x in range(k, j + 1):
        body = (cl[x] - o[x]) * direction
        rng = hi[x] - lo[x]
        a = atr[x]
        if body <= 0 or rng <= 0 or not a or math.isnan(a):
            continue
        if body >= C.DISP_BODY_ATR * a and body / rng >= C.DISP_BODY_RANGE and body > best_body:
            best_d, best_body = x, body
    if best_d is None:
        return None
    vol = df15["volume"].to_numpy(float)
    base = vol[max(0, best_d - 20):best_d]
    rv = float(vol[best_d] / base.mean()) if len(base) and base.mean() > 0 else 1.0
    return {"ref_price": ref["price"], "ref_ts": ref["t"], "break_i": j, "break_ts": int(t[j]),
            "disp_i": best_d, "disp_ts": int(t[best_d]), "disp_atr": float(best_body / atr[best_d]),
            "disp_body_ratio": float(best_body / (hi[best_d] - lo[best_d])), "rv": rv,
            "age_bars": n - 1 - j}


def find_zones(df: Candle, atr: np.ndarray, k: int, d: int, j: int, direction: int) -> Dict[str, Any]:
    """FVG / OB created in the leg [k..j] (displacement candle d)."""
    o, hi, lo, cl = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    t = df["t"].to_numpy(np.int64)
    fvg = ob = None
    for i in range(min(j + 1, len(df) - 1), max(k, 2) - 1, -1):   # i = third candle; middle candle is the displacement
        a = atr[i]
        if direction == 1 and lo[i] > hi[i - 2] and (lo[i] - hi[i - 2]) >= 0.1 * a:
            fvg = {"lo": float(hi[i - 2]), "hi": float(lo[i]), "ts": int(t[i - 1])}
            break
        if direction == -1 and hi[i] < lo[i - 2] and (lo[i - 2] - hi[i]) >= 0.1 * a:
            fvg = {"lo": float(hi[i]), "hi": float(lo[i - 2]), "ts": int(t[i - 1])}
            break
    for x in range(d - 1, max(k - 3, 0) - 1, -1):
        if (direction == 1 and cl[x] < o[x]) or (direction == -1 and cl[x] > o[x]):
            ob = {"lo": float(lo[x]), "hi": float(hi[x]), "ts": int(t[x])}
            break
    return {"fvg": fvg, "ob": ob}


def _overlap(a: Dict[str, float], b: Dict[str, float]) -> Optional[Dict[str, float]]:
    lo, hi = max(a["lo"], b["lo"]), min(a["hi"], b["hi"])
    return {"lo": lo, "hi": hi} if hi > lo else None


def _entry_plan(direction: int, zone: Dict[str, float], price: float) -> Tuple[str, float]:
    """(entry_type, entry). CMP only if live price is inside the zone; else LIMIT at zone midpoint."""
    mid = (zone["lo"] + zone["hi"]) / 2.0
    if zone["lo"] <= price <= zone["hi"]:
        return "CMP", price
    return "LIMIT", mid


# ─────────────────────────── targets & SL ───────────────────────────
def build_targets(direction: int, entry: float, atr15: float, dfs: Dict[str, Candle], price: float) -> List[Dict[str, Any]]:
    side = "H" if direction == 1 else "L"
    lv: List[Dict[str, Any]] = []
    for tf in ("15m", "1h", "4h"):
        if tf in dfs:
            lv += [x for x in tf_levels(dfs[tf], tf, side, price)]
    for tf in ("1h", "4h"):
        if tf in dfs:
            a = atr_series(dfs[tf])[-1]
            dr = dealing_range(find_swings(dfs[tf], C.SWING_LR[tf], C.SWING_LR[tf]), a)
            if dr:
                lv.append({"price": dr["hi"] if direction == 1 else dr["lo"], "src": f"{tf}:range", "tf": tf})
    if "1h" in dfs:                                   # previous UTC day high / low
        h = dfs["1h"]
        day = (h["t"] // 86_400_000)
        prev = h[day == day.iloc[-1] - 1]
        if len(prev) >= 12:
            lv.append({"price": float(prev["high"].max() if direction == 1 else prev["low"].min()),
                       "src": "PDH" if direction == 1 else "PDL", "tf": "1h"})
    lv = [x for x in lv if (x["price"] - entry) * direction > 0]
    lv.sort(key=lambda x: (x["price"] - entry) * direction)
    merged: List[Dict[str, Any]] = []
    for x in lv:
        if merged and abs(x["price"] - merged[-1]["price"]) < 0.25 * atr15:
            continue
        merged.append(x)
    tps: List[Dict[str, Any]] = []
    for x in merged:
        dist = (x["price"] - entry) * direction
        if not tps:
            if dist >= 1.0 * atr15:
                tps.append(x)
        elif (x["price"] - tps[-1]["price"]) * direction >= 0.5 * atr15:
            tps.append(x)
        if len(tps) == 3:
            break
    return tps


def sl_for(direction: int, entry: float, wick: float, atr15: float) -> float:
    sl = wick - direction * C.SL_ATR_BUFFER * atr15
    far = entry - direction * C.SL_MIN_ATR * atr15
    return min(sl, far) if direction == 1 else max(sl, far)


# ─────────────────────────── scoring (heuristic, NOT a probability) ───────────────────────────
SCORE_WEIGHTS = {"context": 25, "location": 15, "liquidity": 15, "trigger": 10, "zone": 10,
                 "refine_5m": 5, "net_rr": 10, "freshness": 5, "volume": 5}      # sums to 100


def score_candidate(c: Dict[str, Any]) -> Dict[str, float]:
    e = c["evidence"]
    s: Dict[str, float] = {}
    s["context"] = 25.0 if c["setup_type"] == "continuation" and e["bias_15m"] == c["dir"] else \
        20.0 if c["setup_type"] == "continuation" else 12.0
    depth = abs(c["entry"] - e["range_1h"]["eq"]) / max((e["range_1h"]["hi"] - e["range_1h"]["lo"]) / 2, 1e-9)
    s["location"] = min(10.0, depth * 10.0) + (5.0 if e.get("pd_15m_ok") else 0.0)
    src_sig = {"1h": 8.0, "15m": 4.0}[e["sweep"]["tf"]]
    s["liquidity"] = src_sig + (3.0 if "eq" in e["sweep"]["src"] else 0.0) + min(4.0, e["sweep"]["pen_atr"] * 4)
    s["trigger"] = min(10.0, e["mss"]["disp_atr"] / 1.5 * 10.0)
    z = e["zone"]
    s["zone"] = (4.0 if z["kind"] == "OB+FVG" else 2.0) + (4.0 if z["touches"] == 0 else 1.0 if z["touches"] == 1 else 0.0) + 2.0
    s["refine_5m"] = {"confirmed": 5.0, "partial": 2.0}.get(c.get("refine_5m", "none"), 0.0)
    s["net_rr"] = max(0.0, min(10.0, (c["net_rr"] - 1.0) / 2.0 * 10.0))
    s["freshness"] = max(0.0, 5.0 * (1 - e["mss"]["age_bars"] / max(C.MSS_MAX_AGE_BARS, 1)))
    s["volume"] = min(5.0, max(0.0, (e["mss"]["rv"] - 1.0) * 5.0))
    return s


def _total(sc: Dict[str, float]) -> float:
    return round(sum(sc.values()), 1)


# ─────────────────────────── per-direction analysis ───────────────────────────
def market_bias(dfs: Dict[str, Candle]) -> Dict[str, int]:
    """Structure bias of 1h and 15m (used for the BTC regime filter)."""
    out = {}
    for tf, key in (("1h", "b1"), ("15m", "b15")):
        d = dfs.get(tf)
        out[key] = structure_bias(d, find_swings(d, C.SWING_LR[tf], C.SWING_LR[tf])) if d is not None and len(d) > 30 else 0
    return out


def _analyze_dir(sym: str, dfs: Dict[str, Candle], price: float, direction: int,
                 btc: Optional[Dict[str, int]] = None) -> Tuple[Optional[Dict[str, Any]], str]:
    """1h = major trend/bias + dealing range. 15m = setup (liquidity sweep, MSS, zone). 5m = entry (refine_5m)."""
    d_name = "LONG" if direction == 1 else "SHORT"
    df15, df1h = dfs["15m"], dfs["1h"]
    atr15s = atr_series(df15)
    atr15 = float(atr15s[-1]) if not math.isnan(atr15s[-1]) else 0.0
    if atr15 <= 0:
        return None, "ATR(15m) unavailable"
    r15 = C.SWING_LR["15m"]
    sw15 = find_swings(df15, r15, r15)
    sw1 = find_swings(df1h, C.SWING_LR["1h"], C.SWING_LR["1h"])
    a1 = atr_series(df1h)[-1]
    bias1, bias15 = structure_bias(df1h, sw1), structure_bias(df15, sw15)
    rng1, rng15 = dealing_range(sw1, a1), dealing_range(sw15, atr15)
    if rng1 is None:
        return None, "no valid 1H dealing range"

    # BTC regime filter (alts follow BTC): block when BTC 1h AND 15m both oppose the trade
    if C.BTC_FILTER and sym != "BTCUSDT" and btc and btc.get("b1") == -direction and btc.get("b15") == -direction:
        return None, f"BTC regime against {d_name} (1h & 15m)"

    # 1) liquidity sweep + reclaim on 15m (levels: 15m and 1h pools)
    side = "L" if direction == 1 else "H"
    levels = tf_levels(df15, "15m", side, price) + tf_levels(df1h, "1h", side, price)
    sweep = find_sweep(df15, atr15s, direction, levels)
    if sweep is None:
        return None, f"no {d_name}-side liquidity sweep+reclaim in last {C.SWEEP_LOOKBACK_BARS} bars"

    # 2) closed-candle MSS with displacement (15m)
    mss = find_mss(df15, atr15s, sw15, sweep, direction)
    if mss is None:
        return None, "no closed-candle MSS with displacement after the sweep"
    if mss["age_bars"] > C.MSS_MAX_AGE_BARS:
        return None, f"MSS stale ({mss['age_bars']} bars)"

    # 3) context: continuation (with 1h trend) vs reversal (optional, stricter)
    pd15_ok = bool(rng15 and ((direction == 1 and price <= rng15["eq"]) or (direction == -1 and price >= rng15["eq"])))
    if bias1 == direction and bias15 != -direction:
        setup_type = "continuation"
    elif C.ALLOW_REVERSAL and sweep["tf"] == "1h" and pd15_ok and mss["disp_atr"] >= 1.0 and mss["rv"] >= 1.2:
        setup_type = "reversal"
    else:
        return None, (f"context invalid: 1H bias {bias1}, 15m bias {bias15} for {d_name} "
                      f"(continuation needs 1H with-trend & 15m not against; reversal "
                      f"{'enabled' if C.ALLOW_REVERSAL else 'disabled'})")

    # 4) zone
    z = find_zones(df15, atr15s, sweep["k"], mss["disp_i"], mss["break_i"], direction)
    fvg, ob = z["fvg"], z["ob"]
    if not fvg and not ob:
        return None, "no FVG/OB formed in the displacement leg"
    ov = _overlap(fvg, ob) if fvg and ob else None
    zone, kind = (ov, "OB+FVG") if ov else ((fvg, "FVG") if fvg else (ob, "OB"))
    cl_after = df15["close"].to_numpy(float)[mss["break_i"] + 1:]
    lows, highs = df15["low"].to_numpy(float)[mss["break_i"] + 1:], df15["high"].to_numpy(float)[mss["break_i"] + 1:]
    if len(cl_after) and ((direction == 1 and cl_after.min() < zone["lo"]) or (direction == -1 and cl_after.max() > zone["hi"])):
        return None, "zone invalidated (closed through)"
    touches = int(((lows <= zone["hi"]) if direction == 1 else (highs >= zone["lo"])).sum()) if len(lows) else 0
    if (direction == 1 and price < zone["lo"]) or (direction == -1 and price > zone["hi"]):
        return None, "live price already through the zone"

    return _finish(sym, dfs, price, direction, atr15, sweep, mss, zone, kind, touches, fvg, ob,
                   setup_type, bias1, bias15, rng1, rng15, pd15_ok)


def _finish(sym, dfs, price, direction, atr15, sweep, mss, zone, kind, touches, fvg, ob,
            setup_type, bias1, bias4, rng1, rng4, pd4_ok, entry_override=None):
    d_name = "LONG" if direction == 1 else "SHORT"
    etype, entry = _entry_plan(direction, zone, price)
    if entry_override:
        etype, entry = entry_override
    dist_atr = abs(entry - price) / atr15
    if etype == "LIMIT" and (abs(entry - price) / price * 100 > C.MAX_ENTRY_DIST_PCT or dist_atr > C.MAX_ENTRY_DIST_ATR):
        return None, f"limit entry too far ({dist_atr:.1f} ATR) — would likely never fill"
    # premium / discount of the ENTRY inside the 1H dealing range
    if (direction == 1 and entry > rng1["eq"]) or (direction == -1 and entry < rng1["eq"]):
        return None, f"entry {entry:.6g} not in {'discount' if direction == 1 else 'premium'} of 1H range (eq {rng1['eq']:.6g})"
    sl = sl_for(direction, entry, sweep["wick"], atr15)
    risk_pct = abs(entry - sl) / entry * 100
    if risk_pct > C.MAX_SL_PCT:
        return None, f"SL distance {risk_pct:.2f}% > MAX_SL_PCT {C.MAX_SL_PCT}%"
    tps = build_targets(direction, entry, atr15, dfs, price)
    if len(tps) < 3:
        return None, f"only {len(tps)} structural target(s) beyond entry (need 3)"
    err = check_levels(d_name, entry, sl, tps[0]["price"], tps[1]["price"], tps[2]["price"])
    if err:
        return None, err
    rr = net_rr(d_name, etype, entry, sl, tps[0]["price"])
    if rr["net_rr"] < C.MIN_NET_RR:
        return None, f"net RR to TP1 {rr['net_rr']:.2f} < {C.MIN_NET_RR} (gross {rr['gross_rr']:.2f})"

    cand = {
        "symbol": sym, "dir": direction, "direction": d_name, "setup_type": setup_type,
        "entry_type": etype, "entry": float(entry), "zone_low": float(zone["lo"]), "zone_high": float(zone["hi"]),
        "sl": float(sl), "tp1": float(tps[0]["price"]), "tp2": float(tps[1]["price"]), "tp3": float(tps[2]["price"]),
        "gross_rr": round(rr["gross_rr"], 2), "net_rr": round(rr["net_rr"], 2), "risk_pct": round(risk_pct, 3),
        "blended_net_rr": round(blended_net_rr(d_name, etype, entry, sl, [t["price"] for t in tps]), 2),
        "price": float(price), "atr15": atr15, "refine_5m": "none",
        "evidence": {
            "bias_15m": bias4, "bias_1h": bias1, "range_1h": rng1, "range_15m": rng4, "pd_15m_ok": pd4_ok,
            "sweep": {**{k: sweep[k] for k in ("level", "src", "tf", "wick", "pen_atr")},
                      "sweep_ts": _iso(sweep["sweep_ts"]), "reclaim_ts": _iso(sweep["reclaim_ts"]),
                      "level_ts": _iso(sweep["level_ts"])},
            "mss": {**{k: mss[k] for k in ("ref_price", "disp_atr", "disp_body_ratio", "rv", "age_bars")},
                    "ref_ts": _iso(mss["ref_ts"]), "break_ts": _iso(mss["break_ts"]), "disp_ts": _iso(mss["disp_ts"])},
            "zone": {"kind": kind, "lo": zone["lo"], "hi": zone["hi"], "touches": touches,
                     "fvg": fvg and {**fvg, "ts": _iso(fvg["ts"])}, "ob": ob and {**ob, "ts": _iso(ob["ts"])}},
            "targets": [{"price": x["price"], "src": x["src"]} for x in tps],
        },
        "_sweep_k_ts": sweep["sweep_ts"], "_wick": sweep["wick"], "_zone": zone,
    }
    cand["score_parts"] = score_candidate(cand)
    cand["score"] = _total(cand["score_parts"])
    return _py(cand), ""


# ─────────────────────────── 5M refinement ───────────────────────────
def refine_5m(cand: Dict[str, Any], dfs: Dict[str, Candle], df5: Optional[Candle], price: float,
              strict: bool = C.REQUIRE_5M) -> Tuple[Optional[Dict[str, Any]], str]:
    """Confirm entry on closed 5M candles AFTER the 15M setup exists. Never passes on missing data in strict mode."""
    if df5 is None or len(df5) < 40:
        if strict:
            return None, "5M data missing/insufficient (strict mode)"
        out = dict(cand); out["refine_5m"] = "missing"; out["reduced_confirmation"] = True
        return out, ""
    d = cand["dir"]
    t5, o, hi, lo, cl = (df5[x].to_numpy(float if x != "t" else np.int64) for x in ("t", "open", "high", "low", "close"))
    a5s = atr_series(df5)
    start = int(np.searchsorted(t5, cand["_sweep_k_ts"]))
    if start >= len(df5) - 3:
        return (None, "5M window too short") if strict else (dict(cand, refine_5m="missing", reduced_confirmation=True), "")
    wick = cand["_wick"]
    seg_cl = cl[start:]
    if (d == 1 and seg_cl.min() < wick) or (d == -1 and seg_cl.max() > wick):
        return None, "5M closed through the sweep extreme (setup invalid)"
    sw5 = find_swings(df5, C.SWING_LR["5m"], C.SWING_LR["5m"])
    ext_i = start + int(np.argmin(lo[start:]) if d == 1 else np.argmax(hi[start:]))
    want = "H" if d == 1 else "L"
    refs = [s for s in sw5 if s["kind"] == want and s["i"] < ext_i]
    mss5 = False
    if refs:
        ref = refs[-1]
        mss5 = any((cl[x] > ref["price"] if d == 1 else cl[x] < ref["price"]) for x in range(ext_i + 1, len(df5)))
    zone15 = {"lo": cand["zone_low"], "hi": cand["zone_high"]}
    z5 = None
    for i in range(len(df5) - 1, max(start, 2) - 1, -1):
        a = a5s[i]
        if math.isnan(a):
            continue
        if d == 1 and lo[i] > hi[i - 2] and lo[i] - hi[i - 2] >= 0.1 * a:
            z5 = {"lo": float(hi[i - 2]), "hi": float(lo[i]), "ts": int(t5[i - 1])}
        elif d == -1 and hi[i] < lo[i - 2] and lo[i - 2] - hi[i] >= 0.1 * a:
            z5 = {"lo": float(hi[i]), "hi": float(lo[i - 2]), "ts": int(t5[i - 1])}
        if z5 and _overlap(z5, zone15):
            break
        z5 = None
    ov = _overlap(z5, zone15) if z5 else None
    if not (mss5 and ov):
        if strict:
            return None, f"5M refinement not confirmed (5M MSS={mss5}, 5M zone inside 15M zone={bool(ov)})"
        out = dict(cand); out["refine_5m"] = "partial" if mss5 else "none"; out["reduced_confirmation"] = True
        return out, ""
    # re-plan entry on the refined (5M ∩ 15M) zone, keep SL on the 15M invalidation
    etype, entry = _entry_plan(d, ov, price)
    res, why = _finish_from_cand(cand, dfs, price, ov, etype, entry)
    if res is None:
        return None, "after 5M refinement: " + why
    res["refine_5m"] = "confirmed"
    res["evidence"]["refine_5m"] = {"zone": {"lo": ov["lo"], "hi": ov["hi"]}, "fvg_5m_ts": _iso(z5["ts"]), "mss_5m": True}
    res["score_parts"] = score_candidate(res)
    res["score"] = _total(res["score_parts"])
    return res, ""


def _finish_from_cand(cand, dfs, price, zone, etype, entry):
    e = cand["evidence"]
    d = cand["dir"]
    cand_new, why = _finish(cand["symbol"], dfs, price, d, cand["atr15"],
                            {"wick": cand["_wick"], "level": e["sweep"]["level"], "src": e["sweep"]["src"],
                             "tf": e["sweep"]["tf"], "pen_atr": e["sweep"]["pen_atr"],
                             "sweep_ts": cand["_sweep_k_ts"], "reclaim_ts": cand["_sweep_k_ts"], "level_ts": cand["_sweep_k_ts"]},
                            {"ref_price": e["mss"]["ref_price"], "disp_atr": e["mss"]["disp_atr"],
                             "disp_body_ratio": e["mss"]["disp_body_ratio"], "rv": e["mss"]["rv"],
                             "age_bars": e["mss"]["age_bars"], "ref_ts": 0, "break_ts": 0, "disp_ts": 0},
                            zone, e["zone"]["kind"], e["zone"]["touches"], e["zone"]["fvg"], e["zone"]["ob"],
                            cand["setup_type"], e["bias_1h"], e["bias_15m"], e["range_1h"], e["range_15m"],
                            e["pd_15m_ok"], entry_override=(etype, entry))
    if cand_new:
        cand_new["evidence"]["sweep"].update({k: e["sweep"][k] for k in ("sweep_ts", "reclaim_ts", "level_ts")})
        cand_new["evidence"]["mss"].update({k: e["mss"][k] for k in ("ref_ts", "break_ts", "disp_ts")})
        cand_new["_sweep_k_ts"] = cand["_sweep_k_ts"]
    return cand_new, why


# ─────────────────────────── public API ───────────────────────────
def analyze_symbol(sym: str, dfs: Dict[str, Candle], price: float,
                   btc: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """Full scalp pipeline for one coin (all in Python): 1h bias → 15m setup → 5m entry confirmation.
    Returns {'candidate': best fully-confirmed setup | None, 'rejects': {direction: exact reason}}."""
    rejects: Dict[str, str] = {}
    for tf, need in (("1h", 60), ("15m", 60), ("5m", 40)):
        if tf not in dfs or dfs[tf] is None or len(dfs[tf]) < need:
            return {"candidate": None, "rejects": {"ALL": f"missing/short {tf} data"}}
    if not (price and price > 0 and math.isfinite(price)):
        return {"candidate": None, "rejects": {"ALL": "invalid live price"}}
    v = dfs["1h"]["volume"].to_numpy(float) * dfs["1h"]["close"].to_numpy(float)   # abnormal-volume artefact
    if len(v) >= 72:
        prior = v[:-24].sum() / max((len(v) - 24) / 24.0, 1.0)
        if prior > 0 and v[-24:].sum() / prior > C.VOLUME_SPIKE_REJECT:
            return {"candidate": None, "rejects": {"ALL": f"abnormal 24h volume spike ({v[-24:].sum() / prior:.1f}x)"}}
    best = None
    for direction in (1, -1):
        name = "LONG" if direction == 1 else "SHORT"
        try:
            cand, why = _analyze_dir(sym, dfs, price, direction, btc)
            if cand is not None:
                cand, why = refine_5m(cand, dfs, dfs.get("5m"), price)
        except Exception as exc:                       # an error is never a confirmation
            cand, why = None, f"engine error: {type(exc).__name__}: {exc}"
        if cand is None:
            rejects[name] = why
        elif best is None or cand["score"] > best["score"]:
            best = cand
    return {"candidate": best, "rejects": rejects}
