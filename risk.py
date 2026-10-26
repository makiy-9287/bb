"""Level sanity + net RR for DISPLAY (fees/slippage/funding). Never used to reject a strategy decision."""
from typing import Dict, Optional

from config import FEE_MAKER, FEE_TAKER, SLIPPAGE_BPS, FUNDING_8H, ASSUMED_HOLD_HOURS


def _finite_pos(*xs) -> bool:
    try:
        return all(x is not None and float(x) == float(x) and abs(float(x)) != float("inf") and float(x) > 0 for x in xs)
    except (TypeError, ValueError):
        return False


def check_levels(direction: str, entry: float, sl: float, tp1: float, tp2: float, tp3: float) -> Optional[str]:
    """None if the numbers can describe a real trade, else the exact reason (format sanity only)."""
    if direction not in ("LONG", "SHORT"):
        return f"bad direction {direction!r}"
    if not _finite_pos(entry, sl, tp1, tp2, tp3):
        return "non-finite or non-positive price"
    if direction == "LONG" and not (sl < entry < tp1 < tp2 < tp3):
        return f"LONG ordering violated sl={sl} entry={entry} tp={tp1}/{tp2}/{tp3}"
    if direction == "SHORT" and not (sl > entry > tp1 > tp2 > tp3):
        return f"SHORT ordering violated sl={sl} entry={entry} tp={tp1}/{tp2}/{tp3}"
    return None


def net_rr(direction: str, entry_type: str, entry: float, sl: float, target: float) -> Dict[str, float]:
    slip = SLIPPAGE_BPS / 10_000.0
    fee_in = FEE_MAKER if entry_type == "LIMIT" else FEE_TAKER
    slip_in = 0.0 if entry_type == "LIMIT" else slip
    fund = (1.0 if direction == "LONG" else -1.0) * FUNDING_8H * ASSUMED_HOLD_HOURS / 8.0
    risk, reward = abs(entry - sl), abs(target - entry)
    if risk <= 0:
        return {"gross_rr": 0.0, "net_rr": 0.0, "risk_pct": 0.0}
    nr = reward - entry * (fee_in + FEE_MAKER + slip_in + fund)
    nk = risk + entry * (fee_in + FEE_TAKER + slip_in + slip + fund)
    return {"gross_rr": reward / risk, "net_rr": nr / nk if nk > 0 else 0.0, "risk_pct": risk / entry * 100.0}
