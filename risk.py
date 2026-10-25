"""Net RR (fees + slippage + funding), SL/TP validation, portfolio exposure limits."""
from typing import Any, Dict, List, Optional, Tuple

from config import (FEE_MAKER, FEE_TAKER, SLIPPAGE_BPS, FUNDING_8H, ASSUMED_HOLD_HOURS,
                    MAX_OPEN_TRADES, MAX_SAME_DIRECTION, TP_SPLIT)


def _finite_pos(*xs) -> bool:
    try:
        return all(x is not None and float(x) == float(x) and abs(float(x)) != float("inf")
                   and float(x) > 0 for x in xs)
    except (TypeError, ValueError):
        return False


def check_levels(direction: str, entry: float, sl: float, tp1: float, tp2: float, tp3: float) -> Optional[str]:
    """None if OK, else the exact rejection reason."""
    if direction not in ("LONG", "SHORT"):
        return f"bad direction {direction!r}"
    if not _finite_pos(entry, sl, tp1, tp2, tp3):
        return "non-finite or non-positive price"
    if direction == "LONG" and not (sl < entry < tp1 < tp2 < tp3):
        return f"LONG ordering violated sl={sl} entry={entry} tp={tp1}/{tp2}/{tp3}"
    if direction == "SHORT" and not (sl > entry > tp1 > tp2 > tp3):
        return f"SHORT ordering violated sl={sl} entry={entry} tp={tp1}/{tp2}/{tp3}"
    return None


def funding_cost_pct(direction: str, hold_hours: float = ASSUMED_HOLD_HOURS,
                     rate_8h: float = FUNDING_8H) -> float:
    """Expected funding cost as a fraction of notional (positive = cost)."""
    sign = 1.0 if direction == "LONG" else -1.0
    return sign * rate_8h * hold_hours / 8.0


def net_rr(direction: str, entry_type: str, entry: float, sl: float, target: float) -> Dict[str, float]:
    """Gross and net RR to `target`. LIMIT entry/targets fill as maker; CMP entry & stops are taker+slippage."""
    slip = SLIPPAGE_BPS / 10_000.0
    fee_in = FEE_MAKER if entry_type == "LIMIT" else FEE_TAKER
    slip_in = 0.0 if entry_type == "LIMIT" else slip
    fund = funding_cost_pct(direction)
    risk = abs(entry - sl)
    reward = abs(target - entry)
    if risk <= 0:
        return {"gross_rr": 0.0, "net_rr": 0.0, "risk_pct": 0.0}
    cost_win = entry * (fee_in + FEE_MAKER + slip_in + fund)
    cost_loss = entry * (fee_in + FEE_TAKER + slip_in + slip + fund)
    net_reward = reward - cost_win
    net_risk = risk + cost_loss
    return {"gross_rr": reward / risk,
            "net_rr": net_reward / net_risk if net_risk > 0 else 0.0,
            "risk_pct": risk / entry * 100.0}


def blended_net_rr(direction, entry_type, entry, sl, tps, split=TP_SPLIT) -> float:
    """Expected net R if all TP legs fill (weights = TP_SPLIT). Informational only."""
    return sum(w * net_rr(direction, entry_type, entry, sl, tp)["net_rr"] for w, tp in zip(split, tps))


def portfolio_allows(direction: str, symbol: str, active: List[Dict[str, Any]]) -> Tuple[bool, str]:
    if any(t.get("symbol") == symbol for t in active):
        return False, f"{symbol} already has an active/pending trade"
    if len(active) >= MAX_OPEN_TRADES:
        return False, f"max open trades reached ({MAX_OPEN_TRADES})"
    same = sum(1 for t in active if str(t.get("direction", "")).upper() == direction)
    if same >= MAX_SAME_DIRECTION:
        return False, f"max same-direction exposure reached ({MAX_SAME_DIRECTION} {direction})"
    return True, ""


def risk_per_trade_note(risk_pct_price: float, account_risk_pct: float) -> Optional[str]:
    """Position size is NOT computed (balance not supplied). Only states the risk relation."""
    if account_risk_pct <= 0 or risk_pct_price <= 0:
        return None
    return (f"Risk {account_risk_pct:.2f}% of account ⇒ notional ≈ {account_risk_pct / risk_pct_price:.2f}× account "
            f"(SL distance {risk_pct_price:.2f}%). Leverage is your choice; liquidation must be beyond SL.")
