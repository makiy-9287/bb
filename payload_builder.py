"""
Agent payload (V11): text OHLCV + chart images, built from the SAME candle window.

Text per coin:
  {"coin":"SOLUSDT",
   "data_4h":[{"T":"10-26,00:00","O":165.45,"H":168.2,"L":164.1,"C":167.35,"V":245120.45}, ... 100],
   "data_1h":[...100], "data_15m":[...100], "data_5m":[...100],
   "ind_last":{"4h":{"RSI14":55.2,"MA20":..,"MA50":..}, ...}}      ← latest indicator values (tiny helper)
T = candle OPEN time, UTC, "MM-DD,HH:MM". CLOSED candles only, oldest → newest.
Images: for every coin 4 charts (4h,1h,15m,5m) showing exactly those candles (+ MA20/MA50, RSI14, volume).
"""
import base64
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import pandas as pd

from chart_render import chart_window, indicator_last
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
    """`frames[tf]` = full fetched history (warm-up included). Candles written = chart_window(...) = the charted ones."""
    out: Dict[str, Any] = {"coin": symbol}
    for tf in TIMEFRAMES:
        out[f"data_{tf}"] = candles(chart_window(frames[tf]))
    out["ind_last"] = {tf: indicator_last(frames[tf]) for tf in TIMEFRAMES}
    return out


def build_group_text(symbols: List[str], frames: Dict[str, Dict[str, pd.DataFrame]], prices: Dict[str, float],
                     now_ms: int) -> str:
    first = frames[symbols[0]]
    last = ", ".join(f"{tf} {fmt_t(int(first[tf]['t'].iloc[-1]))}" for tf in TIMEFRAMES)
    live = ", ".join(f"{s}={_p(prices[s])}" for s in symbols)
    header = (f"Data generated (UTC): {fmt_t(now_ms)}. Candle open times are UTC. Last CLOSED candle per timeframe → {last}.\n"
              f"Live prices now: {live}\n"
              f"Below: exact OHLCV text data (the charts that follow show exactly these candles). Take ALL prices from this data.\n"
              f"COINS:\n")
    body = json.dumps([coin_payload(s, frames[s]) for s in symbols], separators=(",", ":"), ensure_ascii=False)
    return header + body


def image_part(png: bytes) -> Dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode("ascii")}}


def build_group_content(text: str, symbols: List[str], charts: Dict[Tuple[str, str], bytes]) -> List[Dict[str, Any]]:
    """OpenAI/OpenRouter multimodal user content: [text, (label, image) × coins × timeframes]."""
    parts: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    for s in symbols:
        for tf in TIMEFRAMES:
            parts.append({"type": "text", "text": f"CHART {s} {tf}:"})
            parts.append(image_part(charts[(s, tf)]))
    return parts
