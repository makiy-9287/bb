"""
Computes all technical indicators and SMC/ICT metrics from OHLCV DataFrames.
Returns a compact dictionary per symbol per timeframe.
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
    """FIX: Safely extract the last value of a pandas Series.
    Returns None if the series is None, empty, or has no valid value.
    This guards against pandas_ta_classic returning None when there
    aren't enough candles for the requested indicator length."""
    if series is None:
        return None
    try:
        if len(series) == 0:
            return None
        val = series.iloc[-1]
        return val
    except Exception:
        return None


def _swing_highs_lows(df: pd.DataFrame) -> pd.DataFrame:
    """Wrapper around smartmoneyconcepts swing detection."""
    ohlc = df[["open", "high", "low", "close"]].copy()
    return smc.swing_highs_lows(ohlc, swing_length=SWING_LENGTH)


# ── core indicator block ─────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute all indicators for a single timeframe DataFrame."""
    if df.empty or len(df) < 30:
        return {}

    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]
    ohlc  = df[["open", "high", "low", "close"]].copy()

    out: Dict[str, Any] = {}

    # ── momentum / trend (FIX: guarded with _last()) ────────────
    out["rsi"] = _safe_round(_last(ta.rsi(close, length=14)), 2)

    ema20_val  = _last(ta.ema(close, length=20))
    ema50_val  = _last(ta.ema(close, length=50))
    ema200_val = _last(ta.ema(close, length=200))

    out["ema20"]  = _safe_round(ema20_val)
    out["ema50"]  = _safe_round(ema50_val)
    out["ema200"] = _safe_round(ema200_val)

    # FIX: ema_trend now handles missing EMA values gracefully
    if ema20_val is not None and ema50_val is not None and ema200_val is not None:
        if ema20_val > ema50_val > ema200_val:
            out["ema_trend"] = "bull"
        elif ema20_val < ema50_val < ema200_val:
            out["ema_trend"] = "bear"
        else:
            out["ema_trend"] = "range"
    elif ema20_val is not None and ema50_val is not None:
        # Partial trend if EMA200 unavailable (new listing)
        out["ema_trend"] = "bull" if ema20_val > ema50_val else "bear"
    else:
        out["ema_trend"] = None

    atr_val = _last(ta.atr(high, low, close, length=14))
    out["atr"]     = _safe_round(atr_val)
    out["atr_pct"] = (
        _safe_round(atr_val / close.iloc[-1] * 100, 2)
        if atr_val is not None and close.iloc[-1] != 0 else None
    )

    # ── volume profile / POC ────────────────────────────────────
    # (unchanged — insert your existing POC code here)
    price_bins = np.linspace(low.min(), high.max(), POC_BINS + 1)
    mid        = (high + low + close) / 3
    bin_idx    = np.digitize(mid, price_bins) - 1
    bin_idx    = np.clip(bin_idx, 0, POC_BINS - 1)
    vol_profile = np.zeros(POC_BINS)
    for i, b in enumerate(bin_idx):
        vol_profile[b] += vol.iloc[i]
    total_vol = vol_profile.sum()
    poc_bin   = int(np.argmax(vol_profile))
    out["poc"] = _safe_round((price_bins[poc_bin] + price_bins[poc_bin + 1]) / 2)
    out["poc_strength"] = (
        _safe_round(vol_profile[poc_bin] / total_vol * 100, 2)
        if total_vol > 0 else None
    )

    # ── premium / discount ──────────────────────────────────────
    # (unchanged — insert your existing P/D code here)
    recent_high = high.tail(50).max()
    recent_low  = low.tail(50).min()
    eq          = (recent_high + recent_low) / 2
    out["premium_discount"] = (
        "premium" if close.iloc[-1] > eq
        else "discount" if close.iloc[-1] < eq
        else "equilibrium"
    )
    out["range_high"]     = _safe_round(recent_high)
    out["range_low"]      = _safe_round(recent_low)
    out["equilibrium"]    = _safe_round(eq)
    out["price_vs_eq_pct"] = _safe_round((close.iloc[-1] - eq) / eq * 100, 2)

    # ── SMC: swing structure ────────────────────────────────────
    # (unchanged — insert your existing swing block here)
    shl = pd.DataFrame()
    try:
        shl = _swing_highs_lows(df)
        levels = shl["Level"].dropna()
        types  = shl["HighLow"].dropna()
        if len(levels) >= 1 and len(types) >= 1:
            last_type = types.iloc[-1]
            out["swing_high"] = _safe_round(levels.iloc[-1]) if last_type == 1 else None
            out["swing_low"]  = _safe_round(levels.iloc[-1]) if last_type == -1 else None
        else:
            out["swing_high"] = out["swing_low"] = None
    except Exception:
        out["swing_high"] = out["swing_low"] = None

    # ── SMC: BOS / CHoCH ────────────────────────────────────────
    # (unchanged — insert your existing BOS/CHoCH block here)
    try:
        bos_choch = smc.bos_choch(ohlc, shl, close_break=True)
        bos_vals   = bos_choch["BOS"].dropna()
        choch_vals = bos_choch["CHOCH"].dropna()
        lvl_vals   = bos_choch["Level"].dropna()
        out["bos"]       = int(bos_vals.iloc[-1])   if not bos_vals.empty   else 0
        out["choch"]     = int(choch_vals.iloc[-1]) if not choch_vals.empty else 0
        out["bos_level"] = _safe_round(lvl_vals.iloc[-1]) if not lvl_vals.empty else None
    except Exception:
        out["bos"] = out["choch"] = 0
        out["bos_level"] = None

    # ── SMC: Order Blocks ───────────────────────────────────────
    # (unchanged — insert your existing OB block here)
    try:
        ob = smc.ob(ohlc, shl, close_mitigation=True)
        ob_valid = ob[ob["OB"] != 0].dropna(subset=["Top", "Bottom"])
        if not ob_valid.empty:
            last_ob = ob_valid.iloc[-1]
            out["ob_type"]   = "bull" if last_ob["OB"] == 1 else "bear"
            out["ob_top"]    = _safe_round(last_ob["Top"])
            out["ob_bottom"] = _safe_round(last_ob["Bottom"])
            pct = last_ob.get("Percentage", None)
            out["ob_strength"] = _safe_round(pct, 2) if pct is not None else None
        else:
            out.update({"ob_type": None, "ob_top": None,
                        "ob_bottom": None, "ob_strength": None})
    except Exception:
        out.update({"ob_type": None, "ob_top": None,
                    "ob_bottom": None, "ob_strength": None})

    # ── SMC: FVG ────────────────────────────────────────────────
    # (unchanged — insert your existing FVG block here)
    try:
        fvg = smc.fvg(ohlc, join_consecutive=False)
        fvg_valid = fvg[fvg["FVG"] != 0].dropna(subset=["Top", "Bottom"])
        if not fvg_valid.empty:
            last_fvg = fvg_valid.iloc[-1]
            out["fvg_type"]   = "bull" if last_fvg["FVG"] == 1 else "bear"
            out["fvg_top"]    = _safe_round(last_fvg["Top"])
            out["fvg_bottom"] = _safe_round(last_fvg["Bottom"])
        else:
            out.update({"fvg_type": None, "fvg_top": None, "fvg_bottom": None})
    except Exception:
        out.update({"fvg_type": None, "fvg_top": None, "fvg_bottom": None})

    # ── SMC: Liquidity ──────────────────────────────────────────
    # (unchanged — insert your existing liquidity block here)
    try:
        liq = smc.liquidity(ohlc, shl, range_percent=LIQUIDITY_RANGE_PCT)
        liq_valid = liq[liq["Liquidity"] != 0].dropna(subset=["Level"])
        if not liq_valid.empty:
            levels = liq_valid["Level"].tail(3).tolist()
            out["liquidity_levels"] = [_safe_round(l) for l in levels]
        else:
            out["liquidity_levels"] = []
    except Exception:
        out["liquidity_levels"] = []

    # ── Recent price action summary ─────────────────────────────
    out["last_close"]      = _safe_round(close.iloc[-1])
    out["last_candle_dir"] = (
        "bull" if close.iloc[-1] > df["open"].iloc[-1] else "bear"
    )
    out["recent_highs"] = [_safe_round(v) for v in high.tail(5).tolist()]
    out["recent_lows"]  = [_safe_round(v) for v in low.tail(5).tolist()]

    return out


def compute_multi_timeframe(
    symbol: str, data: Dict[str, pd.DataFrame]
) -> Dict[str, Any]:
    """Compute indicators for all timeframes of one symbol.
    FIX: Individual timeframe failures no longer kill the whole symbol."""
    result = {"symbol": symbol}
    for tf, df in data.items():
        try:
            result[tf] = compute_indicators(df)
        except Exception as exc:
            logger.warning("TF failure %s %s: %s", symbol, tf, exc)
            result[tf] = {}
    return result
