"""
Reproducible walk-forward backtest of the DETERMINISTIC engine (the LLM review is NOT simulated).

 • no look-ahead: at step i only closed candles [0..i] are visible (1h/15m are resampled from them);
 • LIMIT orders fill only on touch AFTER signal time, same conservative TP/SL rules as the live monitor
   (same state machine), fees + slippage + funding from config;
 • out-of-sample: the series is split by time (default 60/40) and each part is reported separately;
 • compares MIN_NET_RR thresholds (default 1.5 vs 2.0).

Usage:
  python backtest.py --csv BTCUSDT_5m.csv [--rr 1.5 2.0] [--step 3] [--oos 0.6] [--out report.json]
      CSV columns: t(ms open time),open,high,low,close,volume — 5m candles, ascending, closed.
  python backtest.py --synthetic            # smoke test on synthetic data (proves the harness, NOT profitability)
"""
import argparse
import json
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

import config as C
from ict_engine import analyze_symbol
from ohlcv import resample_closed, clean_ohlcv
from performance import summarize
from trade_monitor import advance_trade, build_trade_record

M5 = 300_000


def run_backtest(df5: pd.DataFrame, symbol: str = "TESTUSDT", step: int = 3, warmup: int = 3200,
                 cooldown_bars: int = 18) -> List[Dict[str, Any]]:
    df5 = df5.reset_index(drop=True)
    trades: List[Dict[str, Any]] = []
    active: Optional[Dict[str, Any]] = None
    last_closed_idx = -10_000
    ptr = 0                                                     # next candle index to feed the open trade
    candles = df5[["t", "open", "high", "low", "close"]].values.tolist()
    for i in range(warmup, len(df5) - 1, step):
        if active is not None:                                  # feed candles up to and including i
            advance_trade(active, candles[ptr:i + 1], step_ms=M5)
            ptr = i + 1
            if active["status"] == "closed":
                trades.append(active)
                last_closed_idx, active = i, None
        if active is not None or i - last_closed_idx < cooldown_bars:
            continue
        s5 = df5.iloc[:i + 1]
        frames = {"5m": s5.tail(100).reset_index(drop=True),
                  "15m": resample_closed(s5.tail(1500), 15).tail(200).reset_index(drop=True),
                  "1h": resample_closed(s5.tail(3500), 60).tail(250).reset_index(drop=True)}
        r = analyze_symbol(symbol, frames, float(s5["close"].iloc[-1]))
        c = r["candidate"]
        if c is None:
            continue
        now_ms = int(s5["t"].iloc[-1]) + M5                      # decision time = close of candle i
        active = build_trade_record({k: v for k, v in c.items() if not k.startswith("_")}, now_ms, step_ms=M5)
        ptr = i + 1
    if active is not None:                                      # still open at the end → not counted (no fake exit)
        active["status"] = "unfinished"
    return trades


def report(trades: List[Dict[str, Any]], split_ms: Optional[int]) -> Dict[str, Any]:
    ins = [t for t in trades if split_ms is None or t["created_ms"] < split_ms]
    oos = [t for t in trades if split_ms is not None and t["created_ms"] >= split_ms]
    return {"in_sample": summarize(ins), "out_of_sample": summarize(oos) if split_ms is not None else None}


def compare_thresholds(df5: pd.DataFrame, thresholds: List[float], oos: float = 0.6, **kw) -> Dict[str, Any]:
    split_ms = int(df5["t"].iloc[int(len(df5) * oos)])
    out = {}
    orig = C.MIN_NET_RR
    try:
        for th in thresholds:
            C.MIN_NET_RR = th
            out[str(th)] = report(run_backtest(df5, **kw), split_ms)
    finally:
        C.MIN_NET_RR = orig
    return out


def _fmt(s: Optional[Dict[str, Any]]) -> str:
    if not s or not s["filled"]:
        return "no filled trades"
    f = lambda x, p=2: "n/a" if x is None else f"{x:.{p}f}"
    return (f"n={s['filled']} win={f(s['win_rate'] and s['win_rate'] * 100, 1)}% exp={f(s['expectancy_r'])}R "
            f"PF={f(s['profit_factor'])} totR={s['total_r']:.2f} maxDD={s['max_drawdown_r']:.2f}R "
            f"fill={f(s['fill_rate'] and s['fill_rate'] * 100, 0)}% amb={s['ambiguous']}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--rr", type=float, nargs="+", default=[1.5, 2.0])
    ap.add_argument("--step", type=int, default=3); ap.add_argument("--oos", type=float, default=0.6)
    ap.add_argument("--out", default="backtest_report.json")
    a = ap.parse_args(argv)
    if a.csv:
        raw = pd.read_csv(a.csv).iloc[:, :6].values.tolist()
        df5, meta = clean_ohlcv(raw, "5m", int(raw[-1][0]) + M5, max_stale_bars=10**9, min_rows=300)
        if df5.empty:
            sys.exit(f"CSV rejected: {meta['reason']}")
        print(f"data: {len(df5)} candles, gaps={meta['gaps']}, dup={meta['duplicates']}, invalid={meta['invalid_rows']}")
        label = a.csv
    elif a.synthetic:
        sys.path.insert(0, "tests")
        import fixtures as F
        df5 = F.split_to_5m(F.make_15m(n=5000, seed=3, pullback=True)); label = "SYNTHETIC"
    else:
        ap.error("give --csv FILE or --synthetic")
    res = compare_thresholds(df5, a.rr, a.oos, step=a.step)
    for th, r in res.items():
        print(f"\nMIN_NET_RR={th}  [{label}]")
        print("  in-sample    :", _fmt(r["in_sample"]))
        print("  out-of-sample:", _fmt(r["out_of_sample"]))
    json.dump(res, open(a.out, "w"), indent=2, default=str)
    print(f"\nwritten {a.out}. Simulated results; small samples prove nothing about profitability.")
    return res


if __name__ == "__main__":
    main()
