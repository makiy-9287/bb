"""
Transforms raw indicator output into a compact JSON payload
optimised for LLM consumption.
"""
import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_KEY_MAP = {
    "symbol": "s",
    "rsi": "rsi",
    "ema20": "e20",
    "ema50": "e50",
    "ema200": "e200",
    "ema_trend": "et",
    "atr": "atr",
    "atr_pct": "atrp",
    "poc": "poc",
    "poc_strength": "pocs",
    "premium_discount": "pd",
    "range_high": "rh",
    "range_low": "rl",
    "equilibrium": "eq",
    "price_vs_eq_pct": "peq",
    "swing_high": "sh",
    "swing_low": "sl",
    "bos": "bos",
    "choch": "choch",
    "bos_level": "bosl",
    "ob_type": "ob",       # FIX: was "obt" which collided with ob_top
    "ob_top": "obT",       # FIX: was "obt_" — ambiguous
    "ob_bottom": "obB",    # FIX: was "obb"
    "ob_strength": "obs",
    "fvg_type": "fvg",     # FIX: was "fvt" which collided with fvg_top
    "fvg_top": "fvgT",     # FIX: was "fvt_" — ambiguous
    "fvg_bottom": "fvgB",  # FIX: was "fvb"
    "liquidity_levels": "liq",
    "last_close": "lc",
    "last_candle_dir": "lcd",
    "recent_highs": "rhi",
    "recent_lows": "rlo",
}


def _compress(d: Dict[str, Any]) -> Dict[str, Any]:
    """Rename keys to short forms and drop None / empty values."""
    out = {}
    for k, v in d.items():
        short = _KEY_MAP.get(k, k)
        if v is None or v == [] or v == {}:
            continue
        out[short] = v
    return out


def build_payload(all_symbol_data: List[Dict[str, Any]]) -> str:
    """Build the final compact JSON string."""
    compact = []
    for sym_data in all_symbol_data:
        entry = {}
        for tf in ("1d", "4h", "1h"):
            tf_data = sym_data.get(tf)
            if tf_data:
                entry[tf] = _compress(tf_data)
        if entry:
            entry["s"] = sym_data["symbol"]
            compact.append(entry)

    payload = {
        "market_context": {
            "total_symbols": len(compact),
            "timeframes": ["1d", "4h", "1h"],
        },
        "data": compact,
    }
    json_str = json.dumps(payload, separators=(",", ":"))
    logger.info("Payload built: %d symbols, %d chars", len(compact), len(json_str))
    return json_str