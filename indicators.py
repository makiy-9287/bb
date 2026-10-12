"""
Technical indicators + extended SMC/ICT metrics.

DETECTIONS:
  • BOS / CHoCH
  • Order Blocks (list of 3) + mitigation
  • Breaker Blocks (former OB broken opposite)
  • FVG (list of 3) + mitigation
  • Liquidity levels (buy-side / sell-side)
  • Liquidity sweeps (wick + close-back confirmation)
  • Equal Highs / Equal Lows (EQH / EQL clusters)
  • Session highs/lows (Asian / London / NY, previous UTC day)
  • Displacement candles (large-body institutional moves)
  • RSI divergence (bullish / bearish)
  • S/R clusters
  • POC / VAH / VAL (volume profile)
  • Premium / Discount / OTE zones

All ta.* calls are guarded with _last() so short-history coins
(newly-listed, tokenized stocks) don't crash the pipeline.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pandas_ta_classic as ta
from smartmoneyconcepts import smc

from config import (
    SWING_LENGTH, LIQUIDITY_RANGE_PCT, POC_BINS,
    SWEEP_LOOKBACK, SWEEP_PEN_ATR_MIN, EQUAL_LEVEL_TOL,
    DISPLACEMENT_ATR_MULT,
)

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════

def _safe_round(v, decimals: int = 4) -> Optional[float]:
    """Round safely; None for NaN/inf/None."""
    try:
        if v is None:
            return None
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, decimals)
    except Exception:
        return None


def _last(series):
    """Safely extract last value; None if None/empty."""
    if series is None:
        return None
    try:
        if len(series) == 0:
            return None
        return series.iloc[-1]
    except Exception:
        return None


def _swing_highs_lows(df: pd.DataFrame) -> pd.DataFrame:
    ohlc = df[["open", "high", "low", "close"]].copy()
    return smc.swing_highs_lows(ohlc, swing_length=SWING_LENGTH)


# ════════════════════════════════════════════════════════════════
#  LIQUIDITY SWEEP DETECTION  (newest-swing-first)
# ════════════════════════════════════════════════════════════════

def _detect_liquidity_sweeps(
    df: pd.DataFrame, shl: pd.DataFrame, atr_val: Optional[float],
) -> Dict[str, Any]:
    """
    Detect recent liquidity sweeps.

    Bullish Sweep (LONG trigger):
      • Candle LOW wicks BELOW a prior swing low
      • Candle CLOSE is back ABOVE that swing low
      • Wick penetration > SWEEP_PEN_ATR_MIN × ATR

    Bearish Sweep (SHORT trigger):
      • Candle HIGH wicks ABOVE a prior swing high
      • Candle CLOSE is back BELOW that swing high
      • Wick penetration > SWEEP_PEN_ATR_MIN × ATR

    Iterates swings NEWEST-first so the reported level is the most
    recently swept one (not a stale older level).
    """
    result: Dict[str, Any] = {"bull": None, "bear": None}

    if (atr_val is None or atr_val <= 0
            or shl is None or shl.empty or df.empty):
        return result

    try:
        n = len(df)
        start = max(n - SWEEP_LOOKBACK, 1)

        swing_lows  = shl[shl["HighLow"] == -1].dropna(subset=["Level"])
        swing_highs = shl[shl["HighLow"] == 1].dropna(subset=["Level"])

        for i in range(n - 1, start - 1, -1):
            candle = df.iloc[i]
            bars_ago = n - 1 - i

            prior_lows  = swing_lows[swing_lows.index < i]
            prior_highs = swing_highs[swing_highs.index < i]

            # ── Bullish sweep — newest swing low first ──────────
            if result["bull"] is None and not prior_lows.empty:
                for idx in range(len(prior_lows) - 1, -1, -1):
                    lvl = float(prior_lows["Level"].iloc[idx])
                    pen = lvl - float(candle["low"])
                    if (pen > SWEEP_PEN_ATR_MIN * atr_val
                            and float(candle["close"]) > lvl):
                        result["bull"] = {
                            "lvl": _safe_round(lvl),
                            "pen": _safe_round(pen / atr_val, 2),
                            "bar": bars_ago,
                        }
                        break

            # ── Bearish sweep — newest swing high first ─────────
            if result["bear"] is None and not prior_highs.empty:
                for idx in range(len(prior_highs) - 1, -1, -1):
                    lvl = float(prior_highs["Level"].iloc[idx])
                    pen = float(candle["high"]) - lvl
                    if (pen > SWEEP_PEN_ATR_MIN * atr_val
                            and float(candle["close"]) < lvl):
                        result["bear"] = {
                            "lvl": _safe_round(lvl),
                            "pen": _safe_round(pen / atr_val, 2),
                            "bar": bars_ago,
                        }
                        break

            if result["bull"] and result["bear"]:
                break

    except Exception as exc:
        logger.debug("Sweep detection failed: %s", exc)

    return result


# ════════════════════════════════════════════════════════════════
#  EQUAL HIGHS / EQUAL LOWS  (EQH / EQL)
# ════════════════════════════════════════════════════════════════

def _detect_equal_levels(shl: pd.DataFrame) -> Dict[str, List[Dict]]:
    """
    Find clusters of 2+ swing points at the same price (EQH/EQL).
    These are high-probability liquidity zones.
    """
    result: Dict[str, List[Dict]] = {"eqh": [], "eql": []}
    if shl is None or shl.empty:
        return result

    try:
        highs = [float(v) for v in
                 shl[shl["HighLow"] == 1]["Level"].dropna().tolist()]
        lows = [float(v) for v in
                shl[shl["HighLow"] == -1]["Level"].dropna().tolist()]

        def _cluster(levels: List[float]) -> List[Dict]:
            used = [False] * len(levels)
            out = []
            for i in range(len(levels)):
                if used[i]:
                    continue
                group = [levels[i]]
                used[i] = True
                for j in range(i + 1, len(levels)):
                    if used[j]:
                        continue
                    if (abs(levels[j] - levels[i]) / levels[i]
                            < EQUAL_LEVEL_TOL):
                        group.append(levels[j])
                        used[j] = True
                if len(group) >= 2:
                    avg = sum(group) / len(group)
                    out.append({"lvl": _safe_round(avg), "n": len(group)})
            return out

        result["eqh"] = _cluster(highs)[:3]
        result["eql"] = _cluster(lows)[:3]
    except Exception as exc:
        logger.debug("EQH/EQL failed: %s", exc)

    return result


# ════════════════════════════════════════════════════════════════
#  SESSION HIGHS / LOWS  (Asia / London / NY, previous UTC day)
# ════════════════════════════════════════════════════════════════

def _detect_session_extremes(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute session highs/lows from the most recent completed UTC day.
      Asia  : 00:00–08:00 UTC
      London: 07:00–16:00 UTC
      NY    : 13:00–21:00 UTC
    """
    out: Dict[str, Any] = {}
    if "ts" not in df.columns or df.empty:
        return out

    try:
        last_ts = df["ts"].iloc[-1]
        prev_day = (last_ts - pd.Timedelta(days=1)).normalize()
        df_prev = df[df["ts"] >= prev_day]
        if df_prev.empty:
            return out

        sessions = {
            "asia":   (0, 8),
            "london": (7, 16),
            "ny":     (13, 21),
        }
        for name, (sh, eh) in sessions.items():
            mask = ((df_prev["ts"].dt.hour >= sh)
                    & (df_prev["ts"].dt.hour < eh))
            sub = df_prev[mask]
            if not sub.empty:
                out[name] = {
                    "h": _safe_round(sub["high"].max()),
                    "l": _safe_round(sub["low"].min()),
                }
    except Exception as exc:
        logger.debug("Session extremes failed: %s", exc)

    return out


# ════════════════════════════════════════════════════════════════
#  DISPLACEMENT CANDLES
# ════════════════════════════════════════════════════════════════

def _detect_displacement(
    df: pd.DataFrame, atr_val: Optional[float],
) -> Dict[str, Any]:
    """
    Find the most recent large-body candle per direction.
    Body > DISPLACEMENT_ATR_MULT × ATR = displacement.
    """
    out: Dict[str, Any] = {"bull": None, "bear": None}
    if atr_val is None or atr_val <= 0 or df.empty:
        return out

    try:
        n = len(df)
        body = (df["close"] - df["open"]).abs()
        threshold = DISPLACEMENT_ATR_MULT * atr_val

        for i in range(n - 1, max(n - 21, 0), -1):
            b = float(body.iloc[i])
            if b < threshold:
                continue
            is_bull = float(df["close"].iloc[i]) > float(df["open"].iloc[i])
            if is_bull and out["bull"] is None:
                out["bull"] = {
                    "body": _safe_round(b / atr_val, 2),
                    "bar": n - 1 - i,
                }
            elif not is_bull and out["bear"] is None:
                out["bear"] = {
                    "body": _safe_round(b / atr_val, 2),
                    "bar": n - 1 - i,
                }
            if out["bull"] and out["bear"]:
                break
    except Exception as exc:
        logger.debug("Displacement failed: %s", exc)

    return out


# ════════════════════════════════════════════════════════════════
#  BREAKER BLOCKS
# ════════════════════════════════════════════════════════════════

def _detect_breaker_blocks(df: pd.DataFrame, ob_df) -> Dict[str, Any]:
    """
    Breaker Block = former OB price has closed through in the opposite
    direction. We scan the most recent OBs first.
    """
    result: Dict[str, Any] = {
        "bb_type": None, "bb_top": None, "bb_bottom": None,
    }
    try:
        if ob_df is None or ob_df.empty or df.empty:
            return result

        valid = ob_df[ob_df["OB"] != 0].dropna(subset=["Top", "Bottom"])
        if valid.empty:
            return result

        closes = df["close"].values
        n = len(closes)

        for i in range(len(valid) - 1, -1, -1):
            row = valid.iloc[i]
            ob_type = int(row["OB"])
            ob_top = float(row["Top"])
            ob_bottom = float(row["Bottom"])

            # Position of this OB inside the original dataframe
            ob_pos = valid.index[i]
            if not isinstance(ob_pos, (int, np.integer)):
                try:
                    ob_pos = int(ob_pos)
                except Exception:
                    ob_pos = i
            if ob_pos >= n or ob_pos < 0:
                continue

            subsequent = closes[ob_pos + 1:]
            if len(subsequent) == 0:
                continue

            if ob_type == 1 and np.any(subsequent < ob_bottom):
                result["bb_type"] = "bear"
                result["bb_top"] = _safe_round(ob_top)
                result["bb_bottom"] = _safe_round(ob_bottom)
                return result
            if ob_type == -1 and np.any(subsequent > ob_top):
                result["bb_type"] = "bull"
                result["bb_top"] = _safe_round(ob_top)
                result["bb_bottom"] = _safe_round(ob_bottom)
                return result

    except Exception as exc:
        logger.debug("BB detection failed: %s", exc)

    return result


# ════════════════════════════════════════════════════════════════
#  S/R CLUSTERING
# ════════════════════════════════════════════════════════════════

def _detect_sr_levels(
    shl: pd.DataFrame, current_price: float,
) -> List[Dict]:
    try:
        if shl is None or shl.empty:
            return []
        swing = shl.dropna(subset=["Level"])
        if swing.empty:
            return []

        points = [
            {"price": float(swing["Level"].iloc[i]),
             "type": "R" if int(swing["HighLow"].iloc[i]) == 1 else "S"}
            for i in range(len(swing))
        ]

        clusters, used = [], set()
        for i, pt in enumerate(points):
            if i in used:
                continue
            group = [pt]
            for j in range(i + 1, len(points)):
                if j in used:
                    continue
                if (abs(points[j]["price"] - pt["price"]) / pt["price"]
                        < 0.005):
                    group.append(points[j])
                    used.add(j)
            avg = sum(g["price"] for g in group) / len(group)
            n_res = sum(1 for g in group if g["type"] == "R")
            n_sup = len(group) - n_res
            clusters.append({
                "p": _safe_round(avg),
                "t": "R" if n_res >= n_sup else "S",
                "n": len(group),
                "dist": _safe_round(
                    abs(avg - current_price) / current_price * 100, 2
                ),
            })
        clusters.sort(key=lambda x: x["dist"])
        return clusters[:5]
    except Exception:
        return []


# ════════════════════════════════════════════════════════════════
#  RSI DIVERGENCE
# ════════════════════════════════════════════════════════════════

def _detect_rsi_divergence(
    df: pd.DataFrame, rsi_series,
) -> Optional[str]:
    try:
        if rsi_series is None or len(rsi_series) < 20:
            return None
        n = len(df)
        highs, lows = [], []
        for i in range(max(1, n - 30), n - 1):
            h, hm, hp = (df["high"].iloc[i],
                         df["high"].iloc[i - 1],
                         df["high"].iloc[i + 1])
            l, lm, lp = (df["low"].iloc[i],
                         df["low"].iloc[i - 1],
                         df["low"].iloc[i + 1])
            if h > hm and h > hp:
                highs.append((i, float(h), float(rsi_series.iloc[i])))
            if l < lm and l < lp:
                lows.append((i, float(l), float(rsi_series.iloc[i])))

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


# ════════════════════════════════════════════════════════════════
#  VALUE AREA  (VAH / VAL, 70% of volume)
# ════════════════════════════════════════════════════════════════

def _compute_value_area(price_bins, vol_profile, total_vol, pct=0.70):
    try:
        if total_vol <= 0:
            return None, None
        order, cum, chosen = np.argsort(vol_profile)[::-1], 0.0, []
        for b in order:
            cum += vol_profile[b]
            chosen.append(int(b))
            if cum >= total_vol * pct:
                break
        hi_bin, lo_bin = max(chosen), min(chosen)
        vah = (price_bins[hi_bin] + price_bins[hi_bin + 1]) / 2
        val = (price_bins[lo_bin] + price_bins[lo_bin + 1]) / 2
        return _safe_round(vah), _safe_round(val)
    except Exception:
        return None, None


# ════════════════════════════════════════════════════════════════
#  MAIN INDICATOR BLOCK
# ════════════════════════════════════════════════════════════════

def compute_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute all indicators for a single timeframe DataFrame."""
    if df.empty or len(df) < 30:
        return {}

    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    ohlc = df[["open", "high", "low", "close"]].copy()
    out: Dict[str, Any] = {}

    # ── momentum / trend ────────────────────────────────────────
    rsi_series = ta.rsi(close, length=14)
    out["rsi"] = _safe_round(_last(rsi_series), 2)

    e20  = _last(ta.ema(close, length=20))
    e50  = _last(ta.ema(close, length=50))
    e200 = _last(ta.ema(close, length=200))
    out["ema20"], out["ema50"], out["ema200"] = (
        _safe_round(e20), _safe_round(e50), _safe_round(e200),
    )
    if e20 is not None and e50 is not None and e200 is not None:
        out["ema_trend"] = ("bull" if e20 > e50 > e200
                            else "bear" if e20 < e50 < e200 else "range")
    elif e20 is not None and e50 is not None:
        out["ema_trend"] = "bull" if e20 > e50 else "bear"
    else:
        out["ema_trend"] = None

    atr_val = _last(ta.atr(high, low, close, length=14))
    out["atr"] = _safe_round(atr_val)
    out["atr_pct"] = (_safe_round(atr_val / close.iloc[-1] * 100, 2)
                      if atr_val and close.iloc[-1] else None)

    # ── volume profile / POC / VAH / VAL ────────────────────────
    price_bins = np.linspace(low.min(), high.max(), POC_BINS + 1)
    mid = (high + low + close) / 3
    bin_idx = np.clip(np.digitize(mid, price_bins) - 1, 0, POC_BINS - 1)
    vol_profile = np.zeros(POC_BINS)
    for i, b in enumerate(bin_idx):
        vol_profile[b] += vol.iloc[i]
    total_vol = vol_profile.sum()
    poc_bin = int(np.argmax(vol_profile))
    out["poc"] = _safe_round(
        (price_bins[poc_bin] + price_bins[poc_bin + 1]) / 2
    )
    out["poc_strength"] = (
        _safe_round(vol_profile[poc_bin] / total_vol * 100, 2)
        if total_vol > 0 else None
    )
    vah, val = _compute_value_area(price_bins, vol_profile, total_vol)
    out["vah"], out["val"] = vah, val

    # ── premium / discount ──────────────────────────────────────
    rh, rl = high.tail(50).max(), low.tail(50).min()
    eq = (rh + rl) / 2
    out["premium_discount"] = (
        "premium" if close.iloc[-1] > eq
        else "discount" if close.iloc[-1] < eq
        else "equilibrium"
    )
    out["range_high"] = _safe_round(rh)
    out["range_low"]  = _safe_round(rl)
    out["equilibrium"] = _safe_round(eq)
    out["price_vs_eq_pct"] = _safe_round(
        (close.iloc[-1] - eq) / eq * 100, 2
    )

    # ── OTE zones ───────────────────────────────────────────────
    rng = rh - rl
    if rng > 0:
        out["ote_long"]  = [_safe_round(rh - rng * 0.790),
                            _safe_round(rh - rng * 0.618)]
        out["ote_short"] = [_safe_round(rl + rng * 0.618),
                            _safe_round(rl + rng * 0.790)]
    else:
        out["ote_long"] = out["ote_short"] = None

    # ── Swing structure ─────────────────────────────────────────
    shl = pd.DataFrame()
    try:
        shl = _swing_highs_lows(df)
        lvls = shl.dropna(subset=["Level"])
        if not lvls.empty:
            shs = lvls[lvls["HighLow"] == 1]["Level"].tail(2).tolist()
            sls = lvls[lvls["HighLow"] == -1]["Level"].tail(2).tolist()
            out["swing_high"]  = _safe_round(shs[-1]) if len(shs) >= 1 else None
            out["swing_low"]   = _safe_round(sls[-1]) if len(sls) >= 1 else None
            out["swing_high2"] = _safe_round(shs[-2]) if len(shs) >= 2 else None
            out["swing_low2"]  = _safe_round(sls[-2]) if len(sls) >= 2 else None
        else:
            for k in ("swing_high", "swing_low", "swing_high2", "swing_low2"):
                out[k] = None
    except Exception:
        for k in ("swing_high", "swing_low", "swing_high2", "swing_low2"):
            out[k] = None

    # ── BOS / CHoCH ─────────────────────────────────────────────
    try:
        bc = smc.bos_choch(ohlc, shl, close_break=True)
        bv = bc["BOS"].dropna()
        cv = bc["CHOCH"].dropna()
        lv = bc["Level"].dropna()
        out["bos"]       = int(bv.iloc[-1]) if not bv.empty else 0
        out["choch"]     = int(cv.iloc[-1]) if not cv.empty else 0
        out["bos_level"] = _safe_round(lv.iloc[-1]) if not lv.empty else None
    except Exception:
        out["bos"] = out["choch"] = 0
        out["bos_level"] = None

    # ── Order Blocks ────────────────────────────────────────────
    ob_df = None
    try:
        ob_df = smc.ob(ohlc, shl, close_mitigation=True)
        ob_valid = ob_df[ob_df["OB"] != 0].dropna(subset=["Top", "Bottom"])
        if not ob_valid.empty:
            last = ob_valid.iloc[-1]
            out["ob_type"]   = "bull" if last["OB"] == 1 else "bear"
            out["ob_top"]    = _safe_round(last["Top"])
            out["ob_bottom"] = _safe_round(last["Bottom"])
            pct = last.get("Percentage")
            out["ob_strength"] = (_safe_round(pct, 2)
                                  if pct is not None else None)
            out["ob_list"] = [
                {"t":   "bull" if ob_valid.iloc[i]["OB"] == 1 else "bear",
                 "top": _safe_round(ob_valid.iloc[i]["Top"]),
                 "bot": _safe_round(ob_valid.iloc[i]["Bottom"])}
                for i in range(len(ob_valid) - 1,
                               max(-1, len(ob_valid) - 4), -1)
            ]
            cp = close.iloc[-1]
            out["ob_mit"] = (
                "yes" if ((last["OB"] == 1 and cp <= last["Top"])
                          or (last["OB"] == -1 and cp >= last["Bottom"]))
                else "no"
            )
        else:
            out.update({"ob_type": None, "ob_top": None, "ob_bottom": None,
                        "ob_strength": None, "ob_list": [], "ob_mit": None})
    except Exception:
        out.update({"ob_type": None, "ob_top": None, "ob_bottom": None,
                    "ob_strength": None, "ob_list": [], "ob_mit": None})

    # ── Breaker Blocks ──────────────────────────────────────────
    out.update(_detect_breaker_blocks(df, ob_df))

    # ── FVG ─────────────────────────────────────────────────────
    try:
        fvg_df = smc.fvg(ohlc, join_consecutive=False)
        fvg_valid = fvg_df[fvg_df["FVG"] != 0].dropna(
            subset=["Top", "Bottom"]
        )
        if not fvg_valid.empty:
            last = fvg_valid.iloc[-1]
            out["fvg_type"]   = "bull" if last["FVG"] == 1 else "bear"
            out["fvg_top"]    = _safe_round(last["Top"])
            out["fvg_bottom"] = _safe_round(last["Bottom"])
            out["fvg_list"] = [
                {"t":   "bull" if fvg_valid.iloc[i]["FVG"] == 1 else "bear",
                 "top": _safe_round(fvg_valid.iloc[i]["Top"]),
                 "bot": _safe_round(fvg_valid.iloc[i]["Bottom"])}
                for i in range(len(fvg_valid) - 1,
                               max(-1, len(fvg_valid) - 4), -1)
            ]
            cp = close.iloc[-1]
            out["fvg_mit"] = (
                "yes" if ((last["FVG"] == 1 and cp <= last["Top"])
                          or (last["FVG"] == -1 and cp >= last["Bottom"]))
                else "no"
            )
        else:
            out.update({"fvg_type": None, "fvg_top": None, "fvg_bottom": None,
                        "fvg_list": [], "fvg_mit": None})
    except Exception:
        out.update({"fvg_type": None, "fvg_top": None, "fvg_bottom": None,
                    "fvg_list": [], "fvg_mit": None})

    # ── Liquidity ───────────────────────────────────────────────
    try:
        liq_df = smc.liquidity(ohlc, shl, range_percent=LIQUIDITY_RANGE_PCT)
        liq_valid = liq_df[liq_df["Liquidity"] != 0].dropna(
            subset=["Level"]
        )
        cp = close.iloc[-1]
        bs, ss, all_lv = [], [], []
        if not liq_valid.empty:
            for i in range(len(liq_valid)):
                lvl = _safe_round(liq_valid["Level"].iloc[i])
                all_lv.append(lvl)
                (bs if lvl > cp else ss).append(lvl)
        out["liquidity_levels"] = all_lv[-3:] if all_lv else []
        out["liq_buy_side"]     = bs[-3:] if bs else []
        out["liq_sell_side"]    = ss[-3:] if ss else []
    except Exception:
        out["liquidity_levels"] = []
        out["liq_buy_side"] = out["liq_sell_side"] = []

    # ── Liquidity Sweeps ────────────────────────────────────────
    sweeps = _detect_liquidity_sweeps(df, shl, atr_val)
    out["sweep_bull"] = sweeps["bull"]
    out["sweep_bear"] = sweeps["bear"]

    # ── Equal Highs / Lows ──────────────────────────────────────
    eq = _detect_equal_levels(shl)
    out["eq_highs"] = eq["eqh"]
    out["eq_lows"]  = eq["eql"]

    # ── Session extremes ────────────────────────────────────────
    out["session_hl"] = _detect_session_extremes(df)

    # ── Displacement ────────────────────────────────────────────
    disp = _detect_displacement(df, atr_val)
    out["disp_bull"] = disp["bull"]
    out["disp_bear"] = disp["bear"]

    # ── S/R + RSI divergence + relative volume ──────────────────
    out["sr_levels"] = _detect_sr_levels(shl, close.iloc[-1])
    out["rsi_div"]   = _detect_rsi_divergence(df, rsi_series)
    try:
        avg20 = vol.tail(20).mean()
        out["rv"] = (_safe_round(vol.iloc[-1] / avg20, 2)
                     if avg20 > 0 else None)
    except Exception:
        out["rv"] = None

    # ── Last price action ───────────────────────────────────────
    out["last_close"]      = _safe_round(close.iloc[-1])
    out["last_candle_dir"] = ("bull" if close.iloc[-1] > df["open"].iloc[-1]
                              else "bear")
    out["recent_highs"]    = [_safe_round(v)
                              for v in high.tail(5).tolist()]
    out["recent_lows"]     = [_safe_round(v)
                              for v in low.tail(5).tolist()]

    return out


# ════════════════════════════════════════════════════════════════
#  MULTI-TIMEFRAME WRAPPER
# ════════════════════════════════════════════════════════════════

def compute_multi_timeframe(
    symbol: str, data: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """Compute indicators for all timeframes of one symbol."""
    result = {"symbol": symbol}
    for tf, df in data.items():
        try:
            result[tf] = compute_indicators(df)
        except Exception as exc:
            logger.warning("TF failure %s %s: %s", symbol, tf, exc)
            result[tf] = {}
    return result
