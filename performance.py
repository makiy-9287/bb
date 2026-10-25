"""Net PnL / R accounting and performance statistics for SIMULATED signal tracking (not exchange fills)."""
from typing import Any, Dict, List

from config import FEE_MAKER, FEE_TAKER, SLIPPAGE_BPS, FUNDING_8H

_FUND_MS = 8 * 3_600_000


def funding_events(t0_ms: int, t1_ms: int) -> int:
    """Number of 00:00/08:00/16:00 UTC funding timestamps in (t0, t1]."""
    if t1_ms <= t0_ms:
        return 0
    return int(t1_ms // _FUND_MS - t0_ms // _FUND_MS)


def compute_result(t: Dict[str, Any]) -> Dict[str, Any]:
    """Net result of a FILLED trade from its exit legs (fractions sum to 1). Costs as fraction of notional."""
    legs = t.get("legs") or []
    fill = t.get("fill_price")
    if not legs or not fill:
        return {"filled": False}
    d = 1.0 if str(t["direction"]).upper() == "LONG" else -1.0
    gross = sum(l["frac"] * d * (l["price"] - fill) / fill for l in legs)
    slip = SLIPPAGE_BPS / 1e4
    fee_in = FEE_MAKER if t.get("entry_type") == "LIMIT" else FEE_TAKER
    fees = fee_in + sum(l["frac"] * (FEE_MAKER if str(l["reason"]).startswith("TP") else FEE_TAKER) for l in legs)
    slippage = (slip if t.get("entry_type") != "LIMIT" else 0.0) + \
        sum(l["frac"] * slip for l in legs if not str(l["reason"]).startswith("TP"))
    closed_ms = int(t["closed_ms"]) if t.get("closed_ms") is not None else int(legs[-1]["ts"])
    filled_ms = int(t["filled_ms"]) if t.get("filled_ms") is not None else closed_ms
    funding = d * FUNDING_8H * funding_events(filled_ms, closed_ms)
    net = gross - fees - slippage - funding
    risk = abs(fill - float(t.get("sl_initial", t["sl"]))) / fill
    return {"filled": True, "gross_pct": gross * 100, "fees_pct": fees * 100, "slippage_pct": slippage * 100,
            "funding_pct": funding * 100, "net_pct": net * 100, "risk_pct": risk * 100,
            "r_multiple": (net / risk) if risk > 0 else 0.0, "win": net > 0}


def summarize(closed: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate stats. Every ratio comes with its sample size; empty samples return None, never a made-up number."""
    rows = sorted(closed, key=lambda x: x.get("closed_ms", 0))
    res = [(t, compute_result(t)) for t in rows]
    filled = [(t, r) for t, r in res if r.get("filled")]
    unfilled = [t for t, r in res if not r.get("filled")]
    limit_total = len(filled) + len(unfilled)
    rs = [r["r_multiple"] for _, r in filled]
    wins = [x for x in rs if x > 0]
    losses = [x for x in rs if x <= 0]
    cum, peak, mdd = 0.0, 0.0, 0.0
    for x in rs:
        cum += x
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
    n = len(filled)
    return {
        "signals_closed": len(rows), "filled": n,
        "unfilled_expired": sum(1 for t in unfilled if t.get("outcome") == "EXPIRED"),
        "unfilled_invalidated": sum(1 for t in unfilled if t.get("outcome") == "INVALIDATED"),
        "fill_rate": (n / limit_total) if limit_total else None,
        "ambiguous": sum(1 for t, _ in filled if t.get("ambiguous")),
        "wins": len(wins), "losses": len(losses),
        "win_rate": (len(wins) / n) if n else None,                       # = wins / filled trades (net > 0)
        "avg_win_r": (sum(wins) / len(wins)) if wins else None,
        "avg_loss_r": (sum(losses) / len(losses)) if losses else None,
        "expectancy_r": (sum(rs) / n) if n else None,
        "profit_factor": (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else None,
        "total_r": sum(rs), "max_drawdown_r": mdd,
        "net_pct_sum": sum(r["net_pct"] for _, r in filled),
        "fees_pct_sum": sum(r["fees_pct"] for _, r in filled),
        "slippage_pct_sum": sum(r["slippage_pct"] for _, r in filled),
        "funding_pct_sum": sum(r["funding_pct"] for _, r in filled),
        "note": "simulated tracking on 1m candles; not exchange fills",
    }
