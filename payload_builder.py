"""
Compact JSON payload builder — text indicators + last 20 raw candles.
"""
import json
import logging
from typing import Any, Dict, List

from config import TIMEFRAMES

logger = logging.getLogger(__name__)

_KEY_MAP = {
    "symbol": "s",
    "rsi": "rsi", "ema20": "e20", "ema50": "e50", "ema200": "e200",
    "ema_trend": "et", "atr": "atr", "atr_pct": "atrp",
    "poc": "poc", "poc_strength": "pocs",
    "premium_discount": "pd", "range_high": "rh", "range_low": "rl",
    "equilibrium": "eq", "price_vs_eq_pct": "peq",
    "swing_high": "sh", "swing_low": "sl",
    "swing_high2": "sh2", "swing_low2": "sl2",
    "bos": "bos", "choch": "choch", "bos_level": "bosl",
    "ob_type": "ob", "ob_top": "obT", "ob_bottom": "obB",
    "ob_strength": "obs", "ob_list": "obL", "ob_mit": "obM",
    "bb_type": "bb", "bb_top": "bbT", "bb_bottom": "bbB",
    "fvg_type": "fvg", "fvg_top": "fvgT", "fvg_bottom": "fvgB",
    "fvg_list": "fvgL", "fvg_mit": "fvgM",
    "liquidity_levels": "liq",
    "liq_buy_side": "liqBS", "liq_sell_side": "liqSS",
    "sweep_bull": "swB", "sweep_bear": "swS",
    "eq_highs": "eqH", "eq_lows": "eqL",
    "session_hl": "ses",
    "disp_bull": "dispB", "disp_bear": "dispS",
    "sr_levels": "sr", "rsi_div": "rsiDiv", "rv": "rv",
    "vah": "vah", "val": "val",
    "ote_long": "oteL", "ote_short": "oteS",
    "last_close": "lc", "last_candle_dir": "lcd",
    "recent_highs": "rhi", "recent_lows": "rlo",
    "raw_candles": "raw",
}


def _compress(d: Dict[str, Any]) -> Dict[str, Any]:
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
        for tf in TIMEFRAMES:
            tf_data = sym_data.get(tf)
            if tf_data:
                entry[tf] = _compress(tf_data)
        if entry:
            entry["s"] = sym_data["symbol"]
            compact.append(entry)

    payload = {
        "market_context": {
            "total_symbols": len(compact),
            "timeframes": TIMEFRAMES,
            "raw_candles_format": "[ts,open,high,low,close,volume] — last 20",
        },
        "data": compact,
    }
    json_str = json.dumps(payload, separators=(",", ":"))
    logger.info("Payload: %d symbols, %d chars, TFs=%s",
                len(compact), len(json_str), TIMEFRAMES)
    return json_str
