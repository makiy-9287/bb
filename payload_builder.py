"""
Transforms raw indicator output into a compact JSON payload.
Now includes all ICT/SMC fields with token-optimised short keys.
"""
import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


_KEY_MAP = {
    # ── existing ──
    "symbol": "s",
    "rsi": "rsi", "ema20": "e20", "ema50": "e50", "ema200": "e200",
    "ema_trend": "et", "atr": "atr", "atr_pct": "atrp",
    "poc": "poc", "poc_strength": "pocs",
    "premium_discount": "pd", "range_high": "rh", "range_low": "rl",
    "equilibrium": "eq", "price_vs_eq_pct": "peq",
    "swing_high": "sh", "swing_low": "sl",
    "bos": "bos", "choch": "choch", "bos_level": "bosl",
    "ob_type": "ob", "ob_top": "obT", "ob_bottom": "obB", "ob_strength": "obs",
    "fvg_type": "fvg", "fvg_top": "fvgT", "fvg_bottom": "fvgB",
    "liquidity_levels": "liq",
    "last_close": "lc", "last_candle_dir": "lcd",
    "recent_highs": "rhi", "recent_lows": "rlo",
    # ── NEW: advanced ICT/SMC fields ──
    "bb_type": "bb", "bb_top": "bbT", "bb_bottom": "bbB",
    "ob_list": "obL", "fvg_list": "fvgL",
    "ob_mit": "obM", "fvg_mit": "fvgM",
    "liq_buy_side": "liqBS", "liq_sell_side": "liqSS",
    "sr_levels": "sr",
    "rsi_div": "rsiDiv",
    "rv": "rv",
    "vah": "vah", "val": "val",
    "ote_long": "oteL", "ote_short": "oteS",
    "swing_high2": "sh2", "swing_low2": "sl2",
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
