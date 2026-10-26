"""
Manual scalp workflow: build ONE file (raw closed candles + engine analysis) for the best 30 coins,
then you upload it to any AI together with SCALP_PROMPT.md.

  python manual_payload.py                  # builds outputs/scalp_payload_<time>.json
  python manual_payload.py --telegram       # …and sends it (and the prompt) to your Telegram admin chat
  python manual_payload.py --serve          # …and serves a download link on your LAN (http://<ip>:8000/<file>)
  Telegram bot command: /payload            # same as --telegram, one tap from the phone

Selection ("best 30" for scalping): liquid active USDT perps (volume + tight spread, from data_fetcher) →
15m ATR% ≥ MANUAL_MIN_ATR15_PCT → ranked by 24h quote volume × 15m ATR% (turnover × movement). BTCUSDT is kept as
regime reference. Candles are CLOSED only (no forming candle), validated by ohlcv.clean_ohlcv.
"""
import argparse
import asyncio
import json
import math
import os
import socket
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import pandas as pd

import config as C
from ict_engine import (analyze_symbol, atr_series, find_swings, structure_bias, dealing_range, tf_levels, market_bias)

OUT_DIR = "outputs"
PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SCALP_PROMPT.md")

LEGEND = {
    "what": "Closed candles only. Each candle = [open,high,low,close,rel_volume] (rel_volume = volume / average of the shown candles); oldest first; last item = most recent CLOSED candle.",
    "tf_end": "meta.last_closed_utc[tf] = open time of the last candle; candle n back is n*step earlier (step: 3m/5m/15m/1h).",
    "analysis": "Computed by deterministic code from the full history, NOT from the trimmed candles shown: b1/b15 = structure bias "
                "1h/15m (+1 bull,-1 bear,0 none); r1 = 1h dealing range [lo,eq,hi], pos1 = price position in it (0=low,1=high; "
                "<0.5 discount, >0.5 premium); liq_up/liq_dn = nearest buy-side/sell-side liquidity [price, source]; "
                "sweep = recent liquidity sweep+reclaim {side, level, wick, bars_ago} (side 'long' = sell-side swept → bullish); "
                "engine = full A-grade setup confirmed by code (entry/SL/TP/net RR/evidence) or null with reject reasons.",
    "costs": "net_rr already includes fees, slippage and funding assumptions (meta.assumptions).",
}


def _r(x: Any) -> Any:
    return float(f"{x:.6g}") if isinstance(x, (int, float)) and math.isfinite(x) else x


def _r5(x: float) -> float:
    return float(f"{x:.5g}")


def _candles(df: pd.DataFrame, n: int) -> List[List[float]]:
    """[o,h,l,c,rv]: prices 5 significant digits; rv = volume / mean volume of the shown candles (1 decimal)."""
    t = df.tail(n)
    mean_v = float(t["volume"].mean()) or 1.0
    return [[_r5(o), _r5(h), _r5(l), _r5(c), round(v / mean_v, 1)]
            for o, h, l, c, v in t[["open", "high", "low", "close", "volume"]].values.tolist()]


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%MZ")


def atr_pct(df: pd.DataFrame) -> Optional[float]:
    a = atr_series(df)[-1]
    c = float(df["close"].iloc[-1])
    return None if math.isnan(a) or c <= 0 else float(a / c * 100)


def select_best(uni: List[Dict[str, Any]], frames15: Dict[str, pd.DataFrame], n: int = C.MANUAL_TOP_N,
                min_atr15_pct: float = C.MANUAL_MIN_ATR15_PCT) -> List[Dict[str, Any]]:
    """Pure & testable: turnover × movement ranking; dead/illiquid coins out; BTC kept as regime reference."""
    rows = []
    for u in uni:
        df = frames15.get(u["symbol"])
        if df is None or len(df) < 60:
            continue
        ap = atr_pct(df)
        if ap is None or (ap < min_atr15_pct and u["symbol"] != "BTCUSDT"):
            continue
        rows.append({**u, "atr15_pct": ap, "scalp_score": u["quote_volume"] * ap})
    rows.sort(key=lambda r: (-r["scalp_score"], r["symbol"]))
    chosen = rows[:n]
    if "BTCUSDT" not in {r["symbol"] for r in chosen}:
        btc = next((r for r in rows if r["symbol"] == "BTCUSDT"), None)
        if btc:
            chosen = chosen[:n - 1] + [btc]
    return chosen


def _levels(dfs: Dict[str, pd.DataFrame], price: float) -> Tuple[List[Any], List[Any]]:
    up, dn = [], []
    for tf in ("15m", "1h"):
        for side, bucket in (("H", up), ("L", dn)):
            for lv in tf_levels(dfs[tf], tf, side, price):
                if (side == "H" and lv["price"] > price) or (side == "L" and lv["price"] < price):
                    bucket.append((abs(lv["price"] - price), _r(lv["price"]), lv["src"]))
    f = lambda b: [[p, s] for _, p, s in sorted(set(b))[:3]]
    return f(up), f(dn)


def _recent_sweep(dfs: Dict[str, pd.DataFrame], price: float) -> Optional[Dict[str, Any]]:
    from ict_engine import find_sweep
    d15 = dfs["15m"]
    atr = atr_series(d15)
    best = None
    for direction, side, name in ((1, "L", "long"), (-1, "H", "short")):
        lv = tf_levels(d15, "15m", side, price) + tf_levels(dfs["1h"], "1h", side, price)
        s = find_sweep(d15, atr, direction, lv)
        if s and (best is None or s["k"] > best["k"]):
            best = {**s, "side": name}
    if not best:
        return None
    return {"side": best["side"], "level": _r(best["level"]), "wick": _r(best["wick"]), "src": best["src"],
            "bars_ago": len(d15) - 1 - best["k"]}


def coin_analysis(sym: str, dfs: Dict[str, pd.DataFrame], price: float, btc: Optional[Dict[str, int]]) -> Dict[str, Any]:
    d1, d15 = dfs["1h"], dfs["15m"]
    sw1 = find_swings(d1, C.SWING_LR["1h"], C.SWING_LR["1h"])
    sw15 = find_swings(d15, C.SWING_LR["15m"], C.SWING_LR["15m"])
    rng = dealing_range(sw1, float(atr_series(d1)[-1]))
    an: Dict[str, Any] = {"b1": structure_bias(d1, sw1), "b15": structure_bias(d15, sw15)}
    if rng:
        an["r1"] = [_r(rng["lo"]), _r(rng["eq"]), _r(rng["hi"])]
        an["pos1"] = round((price - rng["lo"]) / (rng["hi"] - rng["lo"]), 2)
    an["liq_up"], an["liq_dn"] = _levels(dfs, price)
    an["sweep"] = _recent_sweep(dfs, price)
    res = analyze_symbol(sym, dfs, price, btc)
    c = res["candidate"]
    if c:
        e = c["evidence"]
        an["engine"] = {"dir": c["direction"], "type": c["setup_type"], "entry_type": c["entry_type"], "entry": _r(c["entry"]),
                        "zone": [_r(c["zone_low"]), _r(c["zone_high"])], "sl": _r(c["sl"]),
                        "tp": [_r(c["tp1"]), _r(c["tp2"]), _r(c["tp3"])], "tp_src": [t["src"] for t in e["targets"]],
                        "net_rr": c["net_rr"], "risk_pct": c["risk_pct"], "score": c["score"],
                        "sweep": {"level": _r(e["sweep"]["level"]), "src": e["sweep"]["src"], "ts": e["sweep"]["sweep_ts"]},
                        "mss": {"ref": _r(e["mss"]["ref_price"]), "ts": e["mss"]["break_ts"], "disp_atr": round(e["mss"]["disp_atr"], 2)},
                        "zone_kind": e["zone"]["kind"], "r5": c["refine_5m"]}
    else:
        an["engine"] = None
        an["reject"] = {k: v.split(" (")[0][:90] for k, v in res["rejects"].items()}
    return an


def assemble(chosen: List[Dict[str, Any]], frames: Dict[str, Dict[str, pd.DataFrame]], prices: Dict[str, float],
             now_ms: int, counts: Dict[str, int] = C.MANUAL_COUNTS) -> Dict[str, Any]:
    """Pure & testable payload builder."""
    btc = market_bias(frames["BTCUSDT"]) if "BTCUSDT" in frames and all(t in frames["BTCUSDT"] for t in ("1h", "15m")) else None
    coins = []
    for u in chosen:
        s = u["symbol"]
        f = frames.get(s, {})
        if not all(tf in f for tf in counts) or s not in prices:
            continue
        px = prices[s]
        entry = {"s": s, "px": _r(px), "vol24h_musd": round(u["quote_volume"] / 1e6, 1), "spread_pct": round(u["spread_pct"], 4),
                 "atr15_pct": round(u["atr15_pct"], 3), "atr5_pct": round(atr_pct(f["5m"]) or 0, 3),
                 "analysis": coin_analysis(s, f, px, btc)}
        entry["c"] = {tf: _candles(f[tf], n) for tf, n in counts.items()}
        coins.append(entry)
    return {
        "meta": {"generated_utc": _iso(now_ms), "mode": "SCALP ONLY", "n_coins": len(coins),
                 "last_closed_utc": {tf: _iso(int(frames[coins[0]["s"]][tf]["t"].iloc[-1])) for tf in counts} if coins else {},
                 "candles_per_tf": counts, "btc_regime": btc,
                 "selection": "liquid perps ranked by 24h volume × 15m ATR%",
                 "assumptions": {"fee_maker": C.FEE_MAKER, "fee_taker": C.FEE_TAKER, "slippage_bps": C.SLIPPAGE_BPS,
                                 "funding_8h": C.FUNDING_8H, "data_validity": "scalp: use within ~10 minutes of generation"}},
        "legend": LEGEND, "coins": coins}


async def build(top_n: int = C.MANUAL_TOP_N, counts: Dict[str, int] = C.MANUAL_COUNTS) -> Dict[str, Any]:
    """Live: Binance → payload dict. Two-stage fetch (15m for 60 coins to rank, then all TFs for the chosen)."""
    from data_fetcher import get_universe, get_server_time_ms, get_last_prices, fetch_bundle
    now = await get_server_time_ms()
    uni = await get_universe(60)
    if "BTCUSDT" not in {u["symbol"] for u in uni}:
        uni += await get_universe(1)                                  # BTC is always the top-volume perp
    f15, _ = await fetch_bundle([u["symbol"] for u in uni], ["15m"], now)
    chosen = select_best(uni, {s: d["15m"] for s, d in f15.items()}, top_n)
    syms = [c["symbol"] for c in chosen]
    frames, _ = await fetch_bundle(syms, ["1h", "15m", "5m", "3m"], now)
    prices = await get_last_prices(syms)
    return assemble(chosen, frames, prices, now, counts)


def write_file(payload: Dict[str, Any], out_dir: str = OUT_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    name = f"scalp_payload_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}Z.json"
    path = os.path.join(out_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"), ensure_ascii=False)
    return path


def est_tokens(payload: Dict[str, Any]) -> int:
    return int(len(json.dumps(payload, separators=(",", ":"), ensure_ascii=False)) / 3)     # conservative


def split_payload(payload: Dict[str, Any], parts: int) -> List[Dict[str, Any]]:
    """Interleaved split (each part gets a mix of the ranking); BTCUSDT is repeated in every part as regime reference."""
    if parts <= 1:
        return [payload]
    coins = payload["coins"]
    btc = next((c for c in coins if c["s"] == "BTCUSDT"), None)
    rest = [c for c in coins if c["s"] != "BTCUSDT"]
    out = []
    for i in range(parts):
        sub = rest[i::parts] + ([btc] if btc else [])
        meta = {**payload["meta"], "n_coins": len(sub), "part": f"{i + 1}/{parts}"}
        out.append({"meta": meta, "legend": payload["legend"], "coins": sub})
    return out


def write_files(payload: Dict[str, Any], out_dir: str = OUT_DIR, parts: Optional[int] = None) -> List[str]:
    n = parts or (1 + est_tokens(payload) // C.MANUAL_MAX_TOKENS_PER_FILE)
    paths = []
    for i, p in enumerate(split_payload(payload, n)):
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        name = f"scalp_payload_{stamp}Z" + (f"_part{i + 1}of{n}" if n > 1 else "") + ".json"
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(p, f, separators=(",", ":"), ensure_ascii=False)
        paths.append(path)
    return paths


def describe(path: str, payload: Dict[str, Any]) -> str:
    kb = os.path.getsize(path) / 1024
    n_eng = sum(1 for c in payload["coins"] if c["analysis"].get("engine"))
    part = f" · part {payload['meta']['part']}" if payload["meta"].get("part") else ""
    return (f"{payload['meta']['n_coins']} coins{part} · {kb:.0f} KB · ≈{int(kb * 1024 / 3):,} tokens (rough) · "
            f"{n_eng} engine-confirmed setup(s) · generated {payload['meta']['generated_utc']}")


async def send_telegram_document(path: str, caption: str, chat_id: Optional[str] = None) -> None:
    chat = chat_id or C.TELEGRAM_ADMIN_ID
    url = f"https://api.telegram.org/bot{C.TELEGRAM_BOT_TOKEN}/sendDocument"
    async with aiohttp.ClientSession() as s:
        form = aiohttp.FormData()
        form.add_field("chat_id", str(chat))
        form.add_field("caption", caption[:1000])
        form.add_field("document", open(path, "rb"), filename=os.path.basename(path))
        try:
            async with s.post(url, data=form, timeout=aiohttp.ClientTimeout(total=60)) as r:
                if r.status != 200:
                    body = (await r.text())[:200].replace(C.TELEGRAM_BOT_TOKEN, "<token>")
                    raise RuntimeError(f"Telegram sendDocument failed {r.status}: {body}")
        except aiohttp.ClientError as exc:
            raise RuntimeError(str(exc).replace(C.TELEGRAM_BOT_TOKEN, "<token>")) from None


def serve(path: str, port: int = 8000) -> None:
    """Serve ONLY this file (and the prompt) on the LAN until Ctrl+C."""
    import http.server
    import functools
    d = os.path.dirname(os.path.abspath(path))
    allowed = {"/" + os.path.basename(path), "/SCALP_PROMPT.md"}

    class H(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path not in allowed:
                self.send_error(404)
                return
            if self.path == "/SCALP_PROMPT.md":
                self.directory = os.path.dirname(PROMPT_FILE)
            super().do_GET()

    ip = "localhost"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
            sk.connect(("10.255.255.255", 1)); ip = sk.getsockname()[0]
    except Exception:
        pass
    print(f"\n🔗 Download: http://{ip}:{port}/{os.path.basename(path)}\n🔗 Prompt  : http://{ip}:{port}/SCALP_PROMPT.md\n(Ctrl+C to stop; reachable only on your network)")
    with http.server.ThreadingHTTPServer(("0.0.0.0", port), functools.partial(H, directory=d)) as srv:
        srv.serve_forever()


async def build_and_save(parts: Optional[int] = None) -> Tuple[List[str], Dict[str, Any]]:
    payload = await build()
    if not payload["coins"]:
        raise RuntimeError("no valid coins (data fetch failed?)")
    return write_files(payload, parts=parts), payload


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--telegram", action="store_true"); ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--parts", type=int, default=None, help="split into N files (default: automatic by size)")
    a = ap.parse_args(argv)
    paths, payload = asyncio.run(build_and_save(a.parts))
    infos = []
    for p in paths:
        sub = json.load(open(p, encoding="utf-8"))
        infos.append(describe(p, sub))
        print("✅", p, "\n  ", infos[-1])
    path = paths[0]
    if a.telegram:
        for p, info in zip(paths, infos):
            asyncio.run(send_telegram_document(p, "📦 Scalp payload — " + info))
        if os.path.exists(PROMPT_FILE):
            asyncio.run(send_telegram_document(PROMPT_FILE, "📝 Prompt (use with the payload)"))
        print("📨 sent to Telegram")
    if a.serve:
        serve(path, a.port)          # serves the first file; for parts use --telegram or copy the others from outputs/


if __name__ == "__main__":
    main()
