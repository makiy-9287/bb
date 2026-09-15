"""
Computes all technical indicators and SMC/ICT metrics from OHLCV DataFrames.
Now includes: Breaker Blocks, multiple FVGs/OBs, S/R clusters, RSI divergence,
relative volume, VAH/VAL, OTE zones, and mitigation status.
"""
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pandas_ta_classic as ta
from smartmoneyconcepts import smc

from config import SWING_LENGTH, LIQUIDITY_RANGE_PCT, POC_BINS

logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────

def _safe_round(v, decimals=4):
    """Round numeric values safely; return None for NaN/inf/None."""
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
    """Safely extract last value of a pandas Series; None if empty/None."""
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


# ── NEW: Breaker Block detection ─────────────────────────────────

def _detect_breaker_blocks(df: pd.DataFrame, ob_df: pd.DataFrame) -> Dict[str, Any]:
    """A Breaker Block is a former Order Block that price has broken
    through in the opposite direction."""
    result = {"bb_type": None, "bb_top": None, "bb_bottom": None}
    try:
        if ob_df is None or ob_df.empty:
            return result
        valid = ob_df[ob_df["OB"] != 0].dropna(subset=["Top", "Bottom"])
        if valid.empty:
            return result

        closes = df["close"].values
        for idx in range(len(valid) - 1, -1, -1):
            row = valid.iloc[idx]
            ob_type = int(row["OB"])
            ob_top = float(row["Top"])
            ob_bottom = float(row["Bottom"])
            ob_idx = valid.index[idx]
            # Look at candles AFTER this OB
            break_idx = None
            if ob_idx + 1 < len(df):
                subsequent = closes[ob_idx + 1:]
                if ob_type == 1 and np.any(subsequent < ob_bottom):
                    break_idx = ob_idx + 1 + int(np.argmax(subsequent < ob_bottom))
                elif ob_type == -1 and np.any(subsequent > ob_top):
                    break_idx = ob_idx + 1 + int(np.argmax(subsequent > ob_top))
            if break_idx is not None:
                result["bb_type"] = "bear" if ob_type == 1 else "bull"
                result["bb_top"] = _safe_round(ob_top)
                result["bb_bottom"] = _safe_round(ob_bottom)
                return result
    except Exception as exc:
        logger.debug("Breaker block detection failed: %s", exc)
    return result


# ── NEW: Support/Resistance clustering ───────────────────────────

def _detect_sr_levels(shl: pd.DataFrame, current_price: float) -> List[Dict]:
    """Cluster swing points into S/R zones (within 0.5%)."""
    try:
        if shl is None or shl.empty:
            return []
        swing = shl.dropna(subset=["Level"])
        if swing.empty:
            return []

        points = []
        for i in range(len(swing)):
            lvl = float(swing["Level"].iloc[i])
            t   = int(swing["HighLow"].iloc[i])
            points.append({"price": lvl, "type": "R" if t == 1 else "S"})

        clusters = []
        used = set()
        for i, pt in enumerate(points):
            if i in used:
                continue
            group = [pt]
            for j in range(i + 1, len(points)):
                if j in used:
                    continue
                if abs(points[j]["price"] - pt["price"]) / pt["price"] < 0.005:
                    group.append(points[j])
                    used.add(j)
            avg = sum(g["price"] for g in group) / len(group)
            n_res = sum(1 for g in group if g["type"] == "R")
            n_sup = len(group) - n_res
            clusters.append({
                "p": _safe_round(avg),
                "t": "R" if n_res >= n_sup else "S",
                "n": len(group),
                "dist": _safe_round(abs(avg - current_price) / current_price * 100, 2),
            })
        # Sort by proximity to current price, take top 5
        clusters.sort(key=lambda x: x["dist"])
        return clusters[:5]
    except Exception:
        return []


# ── NEW: RSI Divergence ──────────────────────────────────────────

def _detect_rsi_divergence(df: pd.DataFrame, rsi_series) -> str | None:
    """Detect the most recent RSI divergence over last ~30 candles."""
    try:
        if rsi_series is None or len(rsi_series) < 20:
            return None
        highs, lows = [], []
        n = len(df)
        for i in range(max(1, n - 30), n - 1):
            if df["high"].iloc[i] > df["high"].iloc[i-1] and df["high"].iloc[i] > df["high"].iloc[i+1]:
                highs.append((i, float(df["high"].iloc[i]), float(rsi_series.iloc[i])))
            if df["low"].iloc[i] < df["low"].iloc[i-1] and df["low"].iloc[i] < df["low"].iloc[i+1]:
                lows.append((i, float(df["low"].iloc[i]), float(rsi_series.iloc[i])))
        # Bearish: price HH + RSI LH
        if len(highs) >= 2:
            a, b = highs[-2], highs[-1]
            if b[1] > a[1] and b[2] < a[2]:
                return "bear"
        # Bullish: price LL + RSI HL
        if len(lows) >= 2:
            a, b = lows[-2], lows[-1]
            if b[1] < a[1] and b[2] > a[2]:
                return "bull"
        return None
    except Exception:
        return None


# ── NEW: VAH / VAL from volume profile ───────────────────────────

def _compute_value_area(price_bins, vol_profile, total_vol, pct=0.70):
    """Return (VAH, VAL) = 70% value area high/low from volume profile."""
    try:
        if total_vol <= 0:
            return None, None
        order = np.argsort(vol_profile)[::-1]
        cum, chosen = 0.0, []
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


# ── Main indicator block ─────────────────────────────────────────

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

    ema20, ema50, ema200 = (ta.ema(close, length=n) for n in (20, 50, 200))
    e20, e50, e200 = _last(ema20), _last(ema50), _last(ema200)
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
    out["atr"]     = _safe_round(atr_val)
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
    out["poc"] = _safe_round((price_bins[poc_bin] + price_bins[poc_bin + 1]) / 2)
    out["poc_strength"] = (_safe_round(vol_profile[poc_bin] / total_vol * 100, 2)
                           if total_vol > 0 else None)
    vah, val = _compute_value_area(price_bins, vol_profile, total_vol)
    out["vah"], out["val"] = vah, val

    # ── premium / discount ──────────────────────────────────────
    recent_high, recent_low = high.tail(50).max(), low.tail(50).min()
    eq = (recent_high + recent_low) / 2
    out["premium_discount"] = ("premium" if close.iloc[-1] > eq
                                else "discount" if close.iloc[-1] < eq
                                else "equilibrium")
    out["range_high"], out["range_low"] = _safe_round(recent_high), _safe_round(recent_low)
    out["equilibrium"] = _safe_round(eq)
    out["price_vs_eq_pct"] = _safe_round((close.iloc[-1] - eq) / eq * 100, 2)

    # ── OTE zones (0.618–0.79 retracement) ──────────────────────
    rng = recent_high - recent_low
    if rng > 0:
        ote_lh = recent_high - rng * 0.618
        ote_ll = recent_high - rng * 0.790
        ote_sl = recent_low  + rng * 0.618
        ote_sh = recent_low  + rng * 0.790
        out["ote_long"]  = [_safe_round(ote_ll), _safe_round(ote_lh)]
        out["ote_short"] = [_safe_round(ote_sl), _safe_round(ote_sh)]
    else:
        out["ote_long"] = out["ote_short"] = None

    # ── Swing structure ─────────────────────────────────────────
    shl = pd.DataFrame()
    try:
        shl = _swing_highs_lows(df)
        lvls = shl.dropna(subset=["Level"])
        if not lvls.empty:
            # Last two swing highs and lows
            shs = lvls[lvls["HighLow"] == 1]["Level"].tail(2).tolist()
            sls = lvls[lvls["HighLow"] == -1]["Level"].tail(2).tolist()
            out["swing_high"]  = _safe_round(shs[-1]) if len(shs) >= 1 else None
            out["swing_low"]   = _safe_round(sls[-1]) if len(sls) >= 1 else None
            out["swing_high2"] = _safe_round(shs[-2]) if len(shs) >= 2 else None
            out["swing_low2"]  = _safe_round(sls[-2]) if len(sls) >= 2 else None
        else:
            out["swing_high"] = out["swing_low"] = None
            out["swing_high2"] = out["swing_low2"] = None
    except Exception:
        out["swing_high"] = out["swing_low"] = None
        out["swing_high2"] = out["swing_low2"] = None

    # ── BOS / CHoCH ─────────────────────────────────────────────
    try:
        bc = smc.bos_choch(ohlc, shl, close_break=True)
        bv = bc["BOS"].dropna()
        cv = bc["CHOCH"].dropna()
        lv = bc["Level"].dropna()
        out["bos"]       = int(bv.iloc[-1])   if not bv.empty else 0
        out["choch"]     = int(cv.iloc[-1])   if not cv.empty else 0
        out["bos_level"] = _safe_round(lv.iloc[-1]) if not lv.empty else None
    except Exception:
        out["bos"] = out["choch"] = 0
        out["bos_level"] = None

    # ── Order Blocks + list ─────────────────────────────────────
    ob_df = None
    try:
        ob_df = smc.ob(ohlc, shl, close_mitigation=True)
        ob_valid = ob_df[ob_df["OB"] != 0].dropna(subset=["Top", "Bottom"])
        if not ob_valid.empty:
            last = ob_valid.iloc[-1]
            out["ob_type"]   = "bull" if last["OB"] == 1 else "bear"
            out["ob_top"]    = _safe_round(last["Top"])
            out["ob_bottom"] = _safe_round(last["Bottom"])
            pct = last.get("Percentage", None)
            out["ob_strength"] = _safe_round(pct, 2) if pct is not None else None
            # Top 3 OB list
            ob_list = []
            for i in range(len(ob_valid) - 1, max(-1, len(ob_valid) - 4), -1):
                r = ob_valid.iloc[i]
                ob_list.append({
                    "t": "bull" if r["OB"] == 1 else "bear",
                    "top": _safe_round(r["Top"]),
                    "bot": _safe_round(r["Bottom"]),
                })
            out["ob_list"] = ob_list
            # Mitigation status
            cp = close.iloc[-1]
            if last["OB"] == 1:
                out["ob_mit"] = "yes" if cp <= last["Top"] else "no"
            else:
                out["ob_mit"] = "yes" if cp >= last["Bottom"] else "no"
        else:
            out.update({"ob_type": None, "ob_top": None, "ob_bottom": None,
                        "ob_strength": None, "ob_list": [], "ob_mit": None})
    except Exception:
        out.update({"ob_type": None, "ob_top": None, "ob_bottom": None,
                    "ob_strength": None, "ob_list": [], "ob_mit": None})

    # ── Breaker Blocks ──────────────────────────────────────────
    if ob_df is not None:
        out.update(_detect_breaker_blocks(df, ob_df))
    else:
        out.update({"bb_type": None, "bb_top": None, "bb_bottom": None})

    # ── FVG + list + mitigation ─────────────────────────────────
    try:
        fvg_df = smc.fvg(ohlc, join_consecutive=False)
        fvg_valid = fvg_df[fvg_df["FVG"] != 0].dropna(subset=["Top", "Bottom"])
        if not fvg_valid.empty:
            last = fvg_valid.iloc[-1]
            out["fvg_type"]   = "bull" if last["FVG"] == 1 else "bear"
            out["fvg_top"]    = _safe_round(last["Top"])
            out["fvg_bottom"] = _safe_round(last["Bottom"])
            fvg_list = []
            for i in range(len(fvg_valid) - 1, max(-1, len(fvg_valid) - 4), -1):
                r = fvg_valid.iloc[i]
                fvg_list.append({
                    "t": "bull" if r["FVG"] == 1 else "bear",
                    "top": _safe_round(r["Top"]),
                    "bot": _safe_round(r["Bottom"]),
                })
            out["fvg_list"] = fvg_list
            cp = close.iloc[-1]
            if last["FVG"] == 1:
                out["fvg_mit"] = "yes" if cp <= last["Top"] else "no"
            else:
                out["fvg_mit"] = "yes" if cp >= last["Bottom"] else "no"
        else:
            out.update({"fvg_type": None, "fvg_top": None, "fvg_bottom": None,
                        "fvg_list": [], "fvg_mit": None})
    except Exception:
        out.update({"fvg_type": None, "fvg_top": None, "fvg_bottom": None,
                    "fvg_list": [], "fvg_mit": None})

    # ── Liquidity: all + buy-side + sell-side ───────────────────
    try:
        liq_df = smc.liquidity(ohlc, shl, range_percent=LIQUIDITY_RANGE_PCT)
        liq_valid = liq_df[liq_df["Liquidity"] != 0].dropna(subset=["Level"])
        cp = close.iloc[-1]
        bs, ss = [], []
        all_lv = []
        if not liq_valid.empty:
            for i in range(len(liq_valid)):
                lvl = _safe_round(liq_valid["Level"].iloc[i])
                all_lv.append(lvl)
                if lvl > cp:
                    bs.append(lvl)
                else:
                    ss.append(lvl)
        out["liquidity_levels"] = all_lv[-3:] if all_lv else []
        out["liq_buy_side"]     = bs[-3:] if bs else []
        out["liq_sell_side"]    = ss[-3:] if ss else []
    except Exception:
        out["liquidity_levels"] = []
        out["liq_buy_side"] = out["liq_sell_side"] = []

    # ── Support/Resistance clusters ─────────────────────────────
    out["sr_levels"] = _detect_sr_levels(shl, close.iloc[-1])

    # ── RSI Divergence ──────────────────────────────────────────
    out["rsi_div"] = _detect_rsi_divergence(df, rsi_series)

    # ── Relative Volume ─────────────────────────────────────────
    try:
        avg20 = vol.tail(20).mean()
        out["rv"] = _safe_round(vol.iloc[-1] / avg20, 2) if avg20 > 0 else None
    except Exception:
        out["rv"] = None

    # ── Last price action summary ───────────────────────────────
    out["last_close"]      = _safe_round(close.iloc[-1])
    out["last_candle_dir"] = "bull" if close.iloc[-1] > df["open"].iloc[-1] else "bear"
    out["recent_highs"]    = [_safe_round(v) for v in high.tail(5).tolist()]
    out["recent_lows"]     = [_safe_round(v) for v in low.tail(5).tolist()]

    return out


def compute_multi_timeframe(symbol: str, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Compute indicators for all timeframes of one symbol."""
    result = {"symbol": symbol}
    for tf, df in data.items():
        try:
            result[tf] = compute_indicators(df)
        except Exception as exc:
            logger.warning("TF failure %s %s: %s", symbol, tf, exc)
            result[tf] = {}
    return result
