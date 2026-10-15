"""
v4 Indicators — lightweight: latest RSI, MAs, ATR + raw candles.
Agent analyses raw candles itself for SMC/ICT structure.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pandas_ta_classic as ta

from config import CANDLE_COUNTS

logger = logging.getLogger(__name__)


def _safe_round(v, decimals: int = 8) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        # Trim trailing zeros via round then float conversion
        return round(f, decimals)
    except Exception:
        return None


def _last(series):
    if series is None:
        return None
    try:
        return series.iloc[-1] if len(series) else None
    except Exception:
        return None


def _extract_candles(df: pd.DataFrame, n: int) -> List[List[float]]:
    """
    Extract last N candles as [[o,h,l,c,v], ...] arrays.
    Oldest first, newest last. No timestamps (kept out to save tokens).
    """
    try:
        if df.empty or len(df) < n:
            n = len(df)
        sub = df.tail(n)
        out = []
        for _, row in sub.iterrows():
            out.append([
                _safe_round(row["open"], 8),
                _safe_round(row["high"], 8),
                _safe_round(row["low"], 8),
                _safe_round(row["close"], 8),
                _safe_round(row["volume"], 2),
            ])
        return out
    except Exception as exc:
        logger.warning("Candle extraction failed: %s", exc)
        return []


def compute_indicators(df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
    """Compute latest indicators + raw candles for one timeframe."""
    if df.empty or len(df) < 30:
        return {}

    close = df["close"]
    high = df["high"]
    low = df["low"]

    out: Dict[str, Any] = {}

    # Latest indicator values
    out["rsi"] = _safe_round(_last(ta.rsi(close, length=14)), 2)
    out["ma7"] = _safe_round(_last(ta.sma(close, length=7)), 8)
    out["ma25"] = _safe_round(_last(ta.sma(close, length=25)), 8)
    out["ma99"] = _safe_round(_last(ta.sma(close, length=99)), 8)
    out["atr"] = _safe_round(_last(ta.atr(high, low, close, length=14)), 8)
    out["last_close"] = _safe_round(close.iloc[-1], 8)

    # Raw candles (main payload)
    n = CANDLE_COUNTS.get(timeframe, 150)
    out["candles"] = _extract_candles(df, n)

    return out


def compute_multi_timeframe(
    symbol: str, data: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    result = {"symbol": symbol}
    for tf, df in data.items():
        try:
            result[tf] = compute_indicators(df, tf)
        except Exception as exc:
            logger.warning("TF failure %s %s: %s", symbol, tf, exc)
            result[tf] = {}
    return result
