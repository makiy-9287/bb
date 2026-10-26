"""
Agent payload — exactly the format requested:

  {"coin":"SOLUSDT",
   "data_4h":[{"T":"10-26,00:00","O":165.45,"H":168.2,"L":164.1,"C":167.35,"V":245120.45}, ...200],
   "data_1h":[... 200], "data_15m":[... 150]}

T = candle OPEN time, UTC, "MM-DD,HH:MM". Candles are CLOSED only, oldest → newest.
A group message = short header (generation time, last closed candle per TF, live prices) + a JSON list of coins.
"""
import json
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd

from config import TIMEFRAMES

KEYS = ("T", "O", "H", "L", "C", "V")


def fmt_t(ms: int) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%m-%d,%H:%M")


def _p(x: float) -> float:
    return float(f"{x:.8g}")


def candles(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return [{"T": fmt_t(t), "O": _p(o), "H": _p(h), "L": _p(l), "C": _p(c), "V": round(float(v), 2)}
            for t, o, h, l, c, v in df[["t", "open", "high", "low", "close", "volume"]].itertuples(index=False)]


def coin_payload(symbol: str, frames: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"coin": symbol}
    for tf in TIMEFRAMES:
        out[f"data_{tf}"] = candles(frames[tf])
    return out


def build_group_text(symbols: List[str], frames: Dict[str, Dict[str, pd.DataFrame]], prices: Dict[str, float],
                     now_ms: int) -> str:
    first = frames[symbols[0]]
    last = ", ".join(f"{tf} {fmt_t(int(first[tf]['t'].iloc[-1]))}" for tf in TIMEFRAMES)
    live = ", ".join(f"{s}={_p(prices[s])}" for s in symbols)
    header = (f"Data generated (UTC): {fmt_t(now_ms)}. Candle open times are UTC. Last CLOSED candle per timeframe → {last}.\n"
              f"Live prices now: {live}\nCOINS:\n")
    body = json.dumps([coin_payload(s, frames[s]) for s in symbols], separators=(",", ":"), ensure_ascii=False)
    return header + body
