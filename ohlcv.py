"""OHLCV hygiene: dedupe, order, validate, drop unclosed candles, detect gaps/staleness.
Timestamps are exchange open-times in **milliseconds**. Nothing is fabricated or forward-filled."""
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

TF_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000,
         "4h": 14_400_000, "1d": 86_400_000}
COLS = ["t", "open", "high", "low", "close", "volume"]


def clean_ohlcv(raw: List[List[float]], tf: str, now_ms: int,
                max_stale_bars: int = 3, min_rows: int = 30) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return (closed, contiguous candles, meta). `meta["ok"]` False → caller must not trade on it."""
    meta: Dict[str, Any] = {"tf": tf, "ok": False, "reason": "", "duplicates": 0, "gaps": 0,
                            "invalid_rows": 0, "dropped_unclosed": False, "stale": False, "rows": 0}
    if tf not in TF_MS:
        meta["reason"] = f"unknown timeframe {tf}"
        return pd.DataFrame(columns=COLS), meta
    step = TF_MS[tf]
    if not raw:
        meta["reason"] = "no data"
        return pd.DataFrame(columns=COLS), meta
    try:
        df = pd.DataFrame(raw).iloc[:, :6]
        df.columns = COLS
        df = df.apply(pd.to_numeric, errors="coerce")
    except Exception as exc:
        meta["reason"] = f"unparseable: {exc}"
        return pd.DataFrame(columns=COLS), meta

    ok = np.isfinite(df[COLS].to_numpy(dtype=float)).all(axis=1)
    ok &= (df[["open", "high", "low", "close"]] > 0).all(axis=1).to_numpy()
    ok &= (df["high"] >= df[["open", "close", "low"]].max(axis=1)).to_numpy()
    ok &= (df["low"] <= df[["open", "close", "high"]].min(axis=1)).to_numpy()
    ok &= (df["volume"] >= 0).to_numpy()
    meta["invalid_rows"] = int((~ok).sum())
    df = df[ok]

    n0 = len(df)
    df = df.drop_duplicates(subset="t", keep="last").sort_values("t")
    meta["duplicates"] = n0 - len(df)
    df["t"] = df["t"].astype("int64")

    # unclosed candle: open_time + tf > now  (exchange clock)
    closed = df["t"] + step <= now_ms
    if (~closed).any():
        meta["dropped_unclosed"] = True
        df = df[closed]
    if df.empty:
        meta["reason"] = "no closed candles"
        return df.reset_index(drop=True), meta

    # keep only the last contiguous run (never bridge a gap)
    d = df["t"].diff().fillna(step).to_numpy()
    bad = np.flatnonzero(d != step)
    meta["gaps"] = int(len(bad))
    if len(bad):
        df = df.iloc[bad[-1]:]
    df = df.reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    meta["rows"] = len(df)

    age_bars = (now_ms - (int(df["t"].iloc[-1]) + step)) / step
    meta["age_bars"] = float(age_bars)
    if age_bars > max_stale_bars:
        meta["stale"] = True
        meta["reason"] = f"stale: last closed candle {age_bars:.1f} bars old"
        return df, meta
    if len(df) < min_rows:
        meta["reason"] = f"insufficient history ({len(df)}<{min_rows})"
        return df, meta
    meta["ok"] = True
    return df, meta


def resample_closed(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Aggregate closed lower-TF candles into complete higher-TF candles (partial buckets are dropped)."""
    step = minutes * 60_000
    base = int(df["t"].diff().dropna().min())
    g = df.copy()
    g["b"] = g["t"] // step
    agg = g.groupby("b").agg(t=("t", "first"), open=("open", "first"), high=("high", "max"), low=("low", "min"),
                             close=("close", "last"), volume=("volume", "sum"), n=("t", "count"))
    return agg[agg["n"] == step // base].drop(columns="n").reset_index(drop=True)
