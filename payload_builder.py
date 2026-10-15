"""
Compact JSON payload for v4 — raw candles + latest indicators.
"""
import json
import logging
from typing import Any, Dict, List

from config import TIMEFRAMES, CANDLE_COUNTS

logger = logging.getLogger(__name__)


def _compress(tf_data: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only fields we want; drop None/empty."""
    if not tf_data:
        return {}
    out = {}
    for k in ("rsi", "ma7", "ma25", "ma99", "atr", "last_close", "candles"):
        v = tf_data.get(k)
        if v is None or v == [] or v == {}:
            continue
        out[k] = v
    return out


def build_payload(all_symbol_data: List[Dict[str, Any]]) -> str:
    compact = []
    for sym_data in all_symbol_data:
        entry = {}
        for tf in TIMEFRAMES:
            tf_data = sym_data.get(tf)
            if tf_data:
                compressed = _compress(tf_data)
                if compressed:
                    entry[tf] = compressed
        if entry:
            entry["s"] = sym_data["symbol"]
            compact.append(entry)

    payload = {
        "meta": {
            "total_symbols": len(compact),
            "timeframes": TIMEFRAMES,
            "candles_per_tf": CANDLE_COUNTS,
            "candle_format": "[open,high,low,close,volume] oldest-first",
        },
        "data": compact,
    }
    json_str = json.dumps(payload, separators=(",", ":"))
    logger.info("Payload: %d symbols, %d chars", len(compact), len(json_str))
    return json_str
