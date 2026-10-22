"""
Enhanced SMC/ICT indicators.
Adds: equal highs/lows, recent sweep (last 5 candles), POC zone,
plus all previous OB/FVG/BOS/CHoCH/liquidity/S-R fields.
"""
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pandas_ta_classic as ta
from smartmoneyconcepts import smc

from config import (
    SWING_LENGTH, LIQUIDITY_RANGE_PCT, POC_BINS, EQ_LEVEL_TOLERANCE,
)

logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────

def _safe_round(v, d: int = 4):
    try:
        if v is None:
            return None
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, d)
    except Exception:
        return None


def _last(series):
    if series is None:
        return None
    try:
        return series.iloc[-1] if len(series) else None
    except Exception:
        return None


def _swing_highs_lows(df: pd.DataFrame) -> pd.DataFrame:
    ohlc = df[["open", "high", "low", "close"]].copy()
    return smc.swing_highs_lows(ohlc, swing_length=SWING_LENGTH)


# ── equal highs / lows (liquidity pools) ─────────────────────────

def _cluster_levels(levels: List[float], tol: float) -> List[Dict]:
    if not levels:
        return []
    s = sorted(levels)
    clusters, cur = [], [s[0]]
    for lvl in s[1:]:
        if abs(lvl - cur[-1]) / max(abs(cur[-1]), 1e-9) <= tol:
            cur.append(lvl)
        else:
            if len(cur) >= 2:
                clusters.append({
                    "p": _safe_round(sum(cur) / len(cur)),
                    "n": len(cur),
                })
            cur = [lvl]
    if len(cur) >= 2:
        clusters.append({"p": _safe_round(sum(cur) / len(cur)), "n": len(cur)})
    return clusters


def _detect_equal_levels(shl: pd.DataFrame) -> tuple:
    try:
        if shl is None or shl.empty:
            return [], []
        highs = shl[shl["HighLow"] == 1]["Level"].dropna().tolist()
        lows  = shl[shl["HighLow"] == -1]["Level"].dropna().tolist()
        eqh = _cluster_levels(highs, EQ_LEVEL_TOLERANCE)
        eql = _cluster_levels(lows, EQ_LEVEL_TOLERANCE)
        # Sort: highs descending (nearest above), lows ascending (nearest below)
        eqh.sort(key=lambda x: x["p"], reverse=True)
        eql.sort(key=lambda x: x["p"])
        return eqh[:5], eql[:5]
    except Exception:
        return [], []


# ── recent liquidity sweep (last N candles) ──────────────────────

def _detect_recent_sweep(
    df: pd.DataFrame, shl: pd.DataFrame, lookback: int = 5
) -> Dict | None:
    try:
        if shl is None or shl.empty:
            return None
        highs = shl[shl["HighLow"] == 1]["Level"].dropna().tolist()[-10:]
        lows  = shl[shl["HighLow"] == -1]["Level"].dropna().tolist()[-10:]
        n = len(df)
        for i in range(n - 1, max(-1, n - lookback - 1), -1):
            row = df.iloc[i]
            for h in highs:
                if row["high"] > h and row["close"] < h:
                    return {
                        "t": "bear",
                        "l": _safe_round(h),
                        "ch": _safe_round(row["high"]),
                    }
            for l in lows:
                if row["low"] < l and row["close"] > l:
                    return {
                        "t": "bull",
                        "l": _safe_round(l),
                        "cl": _safe_round(row["low"]),
                    }
        return None
    except Exception:
        return None


# ── S/R clustering ───────────────────────────────────────────────

def _detect_sr(shl: pd.DataFrame, price: float) -> List[Dict]:
    try:
        if shl is None or shl.empty:
            return []
        sw = shl.dropna(subset=["Level"])
        pts = [
            {"price": float(sw["Level"].iloc[i]),
             "type": "R" if int(sw["HighLow"].iloc[i]) == 1 else "S"}
            for i in range(len(sw))
        ]
        clusters, used = [], set()
        for i, pt in enumerate(pts):
            if i in used:
                continue
            grp = [pt]
            for j in range(i + 1, len(pts)):
                if j in used:
                    continue
                if abs(pts[j]["price"] - pt["price"]) / pt["price"] < 0.005:
                    grp.append(pts[j])
                    used.add(j)
            avg = sum(g["price"] for g in grp) / len(grp)
            nr = sum(1 for g in grp if g["type"] == "R")
            ns = len(grp) - nr
            clusters.append({
                "p": _safe_round(avg),
                "t": "R" if nr >= ns else "S",
                "n": len(grp),
                "dist": _safe_round(abs(avg - price) / price * 100, 2),
            })
        clusters.sort(key=lambda x: x["dist"])
        return clusters[:5]
    except Exception:
        return []


# ── RSI divergence ───────────────────────────────────────────────

def _rsi_divergence(df: pd.DataFrame, rsi) -> str | None:
    try:
        if rsi is None or len(rsi) < 20:
            return None
        n = len(df)
        highs, lows = [], []
        for i in range(max(1, n - 30), n - 1):
            h, hm, hp = (df["high"].iloc[i], df["high"].iloc[i-1],
                         df["high"].iloc[i+1])
            l, lm, lp = (df["low"].iloc[i], df["low"].iloc[i-1],
                         df["low"].iloc[i+1])
            if h > hm and h > hp:
                highs.append((i, float(h), float(rsi.iloc[i])))
            if l < lm and l < lp:
                lows.append((i, float(l), float(rsi.iloc[i])))
        if len(highs) >= 2:
            a, b = highs[-2], highs[-1]
            if b[1] > a[1] and b[2] < a[2]:
                return "bear"
        if len(lows) >= 2:
            a, b = lows[-2], lows[-1]
            if b[1] < a[1] and b[2] > a[2]:
                return "bull"
        return None
    except Exception:
        return None


# ── Value Area ───────────────────────────────────────────────────

def _value_area(price_bins, vol_prof, total, pct=0.70):
    try:
        if total <= 0:
            return None, None
        order, cum, chosen = np.argsort(vol_prof)[::-1], 0.0, []
        for b in order:
            cum += vol_prof[b]
            chosen.append(int(b))
            if cum >= total * pct:
                break
        hi, lo = max(chosen), min(chosen)
        vah = (price_bins[hi] + price_bins[hi + 1]) / 2
        val = (price_bins[lo] + price_bins[lo + 1]) / 2
        return _safe_round(vah), _safe_round(val)
    except Exception:
        return None, None


# ── Breaker Blocks ───────────────────────────────────────────────

def _breaker_blocks(df: pd.DataFrame, ob_df) -> Dict:
    out = {"bb_type": None, "bb_top": None, "bb_bottom": None}
    try:
        if ob_df is None or ob_df.empty:
            return out
        valid = ob_df[ob_df["OB"] != 0].dropna(subset=["Top", "Bottom"])
        if valid.empty:
            return out
        closes = df["close"].values
        n = len(closes)
        for i in range(len(valid) - 1, -1, -1):
            row = valid.iloc[i]
            ob_type = int(row["OB"])
            top, bot = float(row["Top"]), float(row["Bottom"])
            pos = valid.index[i]
            if not isinstance(pos, (int, np.integer)) or pos >= n:
                continue
            sub = closes[pos + 1:]
            if len(sub) == 0:
                continue
            if ob_type == 1 and np.any(sub < bot):
                return {"bb_type": "bear", "bb_top": _safe_round(top),
                        "bb_bottom": _safe_round(bot)}
            if ob_type == -1 and np.any(sub > top):
                return {"bb_type": "bull", "bb_top": _safe_round(top),
                        "bb_bottom": _safe_round(bot)}
    except Exception:
        pass
    return out


# ── main ─────────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty or len(df) < 30:
        return {}

    close, high, low, vol = (df["close"], df["high"],
                              df["low"], df["volume"])
    ohlc = df[["open", "high", "low", "close"]].copy()
    out: Dict[str, Any] = {}

    # ── momentum / trend ────────────────────────────────────────
    rsi = ta.rsi(close, length=14)
    out["rsi"] = _safe_round(_last(rsi), 2)

    e20 = _last(ta.ema(close, length=20))
    e50 = _last(ta.ema(close, length=50))
    e200 = _last(ta.ema(close, length=200))
    out["ema20"] = _safe_round(e20)
    out["ema50"] = _safe_round(e50)
    out["ema200"] = _safe_round(e200)

    if e20 and e50 and e200:
        out["ema_trend"] = ("bull" if e20 > e50 > e200
                            else "bear" if e20 < e50 < e200 else "range")
    elif e20 and e50:
        out["ema_trend"] = "bull" if e20 > e50 else "bear"
    else:
        out["ema_trend"] = None

    atr_val = _last(ta.atr(high, low, close, length=14))
    out["atr"] = _safe_round(atr_val)
    out["atr_pct"] = (_safe_round(atr_val / close.iloc[-1] * 100, 2)
                      if atr_val and close.iloc[-1] else None)

    # ── volume profile / POC zone ───────────────────────────────
    price_bins = np.linspace(low.min(), high.max(), POC_BINS + 1)
    mid = (high + low + close) / 3
    bin_idx = np.clip(np.digitize(mid, price_bins) - 1, 0, POC_BINS - 1)
    vp = np.zeros(POC_BINS)
    for i, b in enumerate(bin_idx):
        vp[b] += vol.iloc[i]
    total_vol = vp.sum()
    poc_bin = int(np.argmax(vp))
    poc_price = (price_bins[poc_bin] + price_bins[poc_bin + 1]) / 2
    out["poc"] = _safe_round(poc_price)
    out["poc_strength"] = (_safe_round(vp[poc_bin] / total_vol * 100, 2)
                           if total_vol > 0 else None)

    # POC zone = bin boundaries + 0.25×ATR buffer
    buf = (atr_val or 0) * 0.25
    poc_low  = price_bins[poc_bin] - buf
    poc_high = price_bins[poc_bin + 1] + buf
    cp = close.iloc[-1]
    out["poc_zone_high"] = _safe_round(poc_high)
    out["poc_zone_low"]  = _safe_round(poc_low)
    out["inside_poc"]    = bool(poc_low <= cp <= poc_high)
    out["poc_distance_pct"] = _safe_round(abs(cp - poc_price) / cp * 100, 2)

    vah, val = _value_area(price_bins, vp, total_vol)
    out["vah"], out["val"] = vah, val

    # ── premium / discount ──────────────────────────────────────
    rh, rl = high.tail(50).max(), low.tail(50).min()
    eq = (rh + rl) / 2
    out["premium_discount"] = ("premium" if cp > eq
                               else "discount" if cp < eq else "equilibrium")
    out["range_high"] = _safe_round(rh)
    out["range_low"]  = _safe_round(rl)
    out["equilibrium"] = _safe_round(eq)
    out["price_vs_eq_pct"] = _safe_round((cp - eq) / eq * 100, 2)

    # ── OTE ─────────────────────────────────────────────────────
    rng = rh - rl
    if rng > 0:
        out["ote_long"]  = [_safe_round(rh - rng * 0.790),
                            _safe_round(rh - rng * 0.618)]
        out["ote_short"] = [_safe_round(rl + rng * 0.618),
                            _safe_round(rl + rng * 0.790)]
    else:
        out["ote_long"] = out["ote_short"] = None

    # ── swing structure ─────────────────────────────────────────
    shl = pd.DataFrame()
    try:
        shl = _swing_highs_lows(df)
        lv = shl.dropna(subset=["Level"])
        if not lv.empty:
            shs = lv[lv["HighLow"] == 1]["Level"].tail(2).tolist()
            sls = lv[lv["HighLow"] == -1]["Level"].tail(2).tolist()
            out["swing_high"]  = _safe_round(shs[-1]) if shs else None
            out["swing_low"]   = _safe_round(sls[-1]) if sls else None
            out["swing_high2"] = _safe_round(shs[-2]) if len(shs) > 1 else None
            out["swing_low2"]  = _safe_round(sls[-2]) if len(sls) > 1 else None
        else:
            for k in ("swing_high", "swing_low",
                      "swing_high2", "swing_low2"):
                out[k] = None
    except Exception:
        for k in ("swing_high", "swing_low", "swing_high2", "swing_low2"):
            out[k] = None

    # ── BOS / CHoCH ─────────────────────────────────────────────
    try:
        bc = smc.bos_choch(ohlc, shl, close_break=True)
        bv, cv, lv = (bc["BOS"].dropna(), bc["CHOCH"].dropna(),
                      bc["Level"].dropna())
        out["bos"]   = int(bv.iloc[-1]) if not bv.empty else 0
        out["choch"] = int(cv.iloc[-1]) if not cv.empty else 0
        out["bos_level"] = _safe_round(lv.iloc[-1]) if not lv.empty else None
    except Exception:
        out["bos"] = out["choch"] = 0
        out["bos_level"] = None

    # ── Order Blocks ────────────────────────────────────────────
    ob_df = None
    try:
        ob_df = smc.ob(ohlc, shl, close_mitigation=True)
        ov = ob_df[ob_df["OB"] != 0].dropna(subset=["Top", "Bottom"])
        if not ov.empty:
            last = ov.iloc[-1]
            out["ob_type"]   = "bull" if last["OB"] == 1 else "bear"
            out["ob_top"]    = _safe_round(last["Top"])
            out["ob_bottom"] = _safe_round(last["Bottom"])
            pct = last.get("Percentage")
            out["ob_strength"] = (_safe_round(pct, 2)
                                  if pct is not None else None)
            out["ob_list"] = [
                {"t": "bull" if ov.iloc[i]["OB"] == 1 else "bear",
                 "top": _safe_round(ov.iloc[i]["Top"]),
                 "bot": _safe_round(ov.iloc[i]["Bottom"])}
                for i in range(len(ov) - 1,
                               max(-1, len(ov) - 4), -1)
            ]
            out["ob_mit"] = ("yes"
                if (last["OB"] == 1 and cp <= last["Top"])
                or (last["OB"] == -1 and cp >= last["Bottom"])
                else "no")
        else:
            out.update({"ob_type": None, "ob_top": None,
                        "ob_bottom": None, "ob_strength": None,
                        "ob_list": [], "ob_mit": None})
    except Exception:
        out.update({"ob_type": None, "ob_top": None,
                    "ob_bottom": None, "ob_strength": None,
                    "ob_list": [], "ob_mit": None})

    out.update(_breaker_blocks(df, ob_df))

    # ── FVG ─────────────────────────────────────────────────────
    try:
        fvg_df = smc.fvg(ohlc, join_consecutive=False)
        fv = fvg_df[fvg_df["FVG"] != 0].dropna(subset=["Top", "Bottom"])
        if not fv.empty:
            last = fv.iloc[-1]
            out["fvg_type"]   = "bull" if last["FVG"] == 1 else "bear"
            out["fvg_top"]    = _safe_round(last["Top"])
            out["fvg_bottom"] = _safe_round(last["Bottom"])
            out["fvg_list"] = [
                {"t": "bull" if fv.iloc[i]["FVG"] == 1 else "bear",
                 "top": _safe_round(fv.iloc[i]["Top"]),
                 "bot": _safe_round(fv.iloc[i]["Bottom"])}
                for i in range(len(fv) - 1,
                               max(-1, len(fv) - 4), -1)
            ]
            out["fvg_mit"] = ("yes"
                if (last["FVG"] == 1 and cp <= last["Top"])
                or (last["FVG"] == -1 and cp >= last["Bottom"])
                else "no")
        else:
            out.update({"fvg_type": None, "fvg_top": None,
                        "fvg_bottom": None, "fvg_list": [],
                        "fvg_mit": None})
    except Exception:
        out.update({"fvg_type": None, "fvg_top": None,
                    "fvg_bottom": None, "fvg_list": [],
                    "fvg_mit": None})

    # ── liquidity levels ────────────────────────────────────────
    try:
        lq = smc.liquidity(ohlc, shl, range_percent=LIQUIDITY_RANGE_PCT)
        lv = lq[lq["Liquidity"] != 0].dropna(subset=["Level"])
        bs, ss, allv = [], [], []
        if not lv.empty:
            for i in range(len(lv)):
                x = _safe_round(lv["Level"].iloc[i])
                allv.append(x)
                (bs if x > cp else ss).append(x)
        out["liquidity_levels"] = allv[-3:] if allv else []
        out["liq_buy_side"]     = bs[-3:] if bs else []
        out["liq_sell_side"]    = ss[-3:] if ss else []
    except Exception:
        out["liquidity_levels"] = []
        out["liq_buy_side"] = out["liq_sell_side"] = []

    # ── equal highs / lows ──────────────────────────────────────
    eqh, eql = _detect_equal_levels(shl)
    out["eq_highs"] = eqh
    out["eq_lows"]  = eql

    # ── recent sweep (last 5 candles) ───────────────────────────
    out["recent_sweep"] = _detect_recent_sweep(df, shl, lookback=5)

    # ── S/R + RSI divergence + rel volume ───────────────────────
    out["sr_levels"] = _detect_sr(shl, cp)
    out["rsi_div"]   = _rsi_divergence(df, rsi)
    try:
        avg20 = vol.tail(20).mean()
        out["rv"] = (_safe_round(vol.iloc[-1] / avg20, 2)
                     if avg20 > 0 else None)
    except Exception:
        out["rv"] = None

    # ── last price action ───────────────────────────────────────
    out["last_close"] = _safe_round(cp)
    out["last_candle_dir"] = ("bull" if cp > df["open"].iloc[-1]
                              else "bear")
    out["recent_highs"] = [_safe_round(v) for v in high.tail(5).tolist()]
    out["recent_lows"]  = [_safe_round(v) for v in low.tail(5).tolist()]

    return out


def compute_multi_timeframe(
    symbol: str, data: Dict[str, pd.DataFrame]
) -> Dict[str, Any]:
    result = {"symbol": symbol}
    for tf, df in data.items():
        try:
            result[tf] = compute_indicators(df)
        except Exception as exc:
            logger.warning("TF failure %s %s: %s", symbol, tf, exc)
            result[tf] = {}
    return result
