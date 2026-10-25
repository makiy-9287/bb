"""Compact payload for the LLM reviewer: only fully-confirmed setups, with the evidence needed to judge them."""
import json
from typing import Any, Dict, List

import pandas as pd


def _r(x: float) -> float:
    return float(f"{x:.6g}")


def _ohlc(df: pd.DataFrame, n: int) -> List[List[float]]:
    return [[_r(a) for a in row] for row in df[["open", "high", "low", "close"]].tail(n).values.tolist()]


def candidate_entry(c: Dict[str, Any], frames: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    e = c["evidence"]
    return {
        "s": c["symbol"], "dir": c["direction"], "type": c["setup_type"], "et": c["entry_type"],
        "e": _r(c["entry"]), "z": [_r(c["zone_low"]), _r(c["zone_high"])], "sl": _r(c["sl"]),
        "tp": [_r(c["tp1"]), _r(c["tp2"]), _r(c["tp3"])],
        "tps": [t["src"] for t in e["targets"]],
        "nrr": c["net_rr"], "risk%": c["risk_pct"], "atr": _r(c["atr15"]), "px": _r(c["price"]),
        "ctx": {"b1": e["bias_1h"], "b15": e["bias_15m"],
                "r1": [_r(e["range_1h"]["lo"]), _r(e["range_1h"]["eq"]), _r(e["range_1h"]["hi"])]},
        "sw": {"level": _r(e["sweep"]["level"]), "src": e["sweep"]["src"], "ts": e["sweep"]["sweep_ts"],
               "wick": _r(e["sweep"]["wick"])},
        "mss": {"ref": _r(e["mss"]["ref_price"]), "ts": e["mss"]["break_ts"], "disp": round(e["mss"]["disp_atr"], 2),
                "rv": round(e["mss"]["rv"], 2), "age": e["mss"]["age_bars"]},
        "zone": {"kind": e["zone"]["kind"], "touches": e["zone"]["touches"]},
        "r5": c.get("refine_5m", "none"),
        "c15": _ohlc(frames["15m"], 12), "c5": _ohlc(frames["5m"], 18),
    }


def build_candidate_payload(cands: List[Dict[str, Any]], frames_by_sym: Dict[str, Dict[str, pd.DataFrame]]) -> str:
    return json.dumps([candidate_entry(c, frames_by_sym[c["symbol"]]) for c in cands],
                      separators=(",", ":"), ensure_ascii=False)
