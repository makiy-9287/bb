"""
Candlestick chart renderer for the vision model (fast, headless, no pyplot global state).

Each chart = last SHOW_N CLOSED candles + volume bars + MA lines + RSI panel, 1536×1024 px
(= 3×2 tiles of 512 px, the native tile size of the Nemotron VL vision encoder → no resampling blur).
Indicators are computed on the FULL fetched history (warm-up), then only the last SHOW_N candles are drawn.
The text payload uses the very same slice (`chart_window`), so chart and numbers always match 1:1.
"""
import io
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection   # noqa: E402
from matplotlib.figure import Figure                  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg        # noqa: E402

from config import SHOW_CANDLES, MA_PERIODS, RSI_LEN, CHART_W, CHART_H, CHART_WORKERS

logger = logging.getLogger(__name__)

UP, DOWN = "#0f9d58", "#d93025"                      # green / red, strong contrast on white
MA_COLORS = ["#1565c0", "#ef6c00", "#6a1b9a"]


# ───────────────────────────── indicators ─────────────────────────────
def sma(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        cs = np.cumsum(np.insert(x, 0, 0.0))
        out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    """Wilder RSI (EMA alpha=1/n) over the full history."""
    out = np.full(len(close), np.nan)
    if len(close) <= n:
        return out
    d = np.diff(close)
    up, dn = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    au, ad = up[:n].mean(), dn[:n].mean()
    out[n] = 100.0 if ad == 0 else 100 - 100 / (1 + au / ad)
    for i in range(n, len(d)):
        au, ad = (au * (n - 1) + up[i]) / n, (ad * (n - 1) + dn[i]) / n
        out[i + 1] = 100.0 if ad == 0 else 100 - 100 / (1 + au / ad)
    return out


def chart_window(df: pd.DataFrame, show_n: int = SHOW_CANDLES) -> pd.DataFrame:
    """THE slice shown in the chart AND written to the text payload."""
    return df.tail(show_n).reset_index(drop=True)


def indicator_last(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    """Latest RSI/MA values (computed with warm-up) — tiny numeric helper for the text payload."""
    c = df["close"].to_numpy(float)
    out: Dict[str, Optional[float]] = {}
    r = rsi(c, RSI_LEN)[-1]
    out[f"RSI{RSI_LEN}"] = None if np.isnan(r) else round(float(r), 1)
    for p in MA_PERIODS:
        m = sma(c, p)[-1]
        out[f"MA{p}"] = None if np.isnan(m) else float(f"{m:.8g}")
    return out


# ───────────────────────────── rendering ─────────────────────────────
def _fmt(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%m-%d %H:%M")


def render_chart(job: Dict[str, Any]) -> bytes:
    """job = {symbol, tf, t, o, h, l, c, v (numpy arrays, FULL history), show_n}. Returns PNG bytes (raises on bad data)."""
    sym, tf, show_n = job["symbol"], job["tf"], job["show_n"]
    t, o, h, l, c, v = (np.asarray(job[k]) for k in ("t", "o", "h", "l", "c", "v"))
    if len(c) < show_n or not np.isfinite(np.concatenate([o, h, l, c])).all():
        raise ValueError(f"{sym} {tf}: not enough/invalid data ({len(c)} candles)")
    mas = [(p, sma(c, p)) for p in MA_PERIODS]
    rs = rsi(c, RSI_LEN)
    s = slice(len(c) - show_n, len(c))
    t, o, h, l, c, v, rs = t[s], o[s], h[s], l[s], c[s], v[s], rs[s]
    x = np.arange(show_n)

    W, H = CHART_W, CHART_H
    fig = Figure(figsize=(W / 100, H / 100), dpi=100, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0.012, 0.335, 0.905, 0.595])             # price
    axr = fig.add_axes([0.012, 0.075, 0.905, 0.205], sharex=ax)  # RSI
    for a in (ax, axr):
        a.set_facecolor("white")
        a.yaxis.tick_right()
        a.grid(True, color="#d9d9d9", lw=0.7)
        a.tick_params(labelsize=13, length=3)
        for sp in a.spines.values():
            sp.set_color("#888888")

    up = c >= o
    colors = np.where(up, UP, DOWN)
    ax.add_collection(LineCollection([[(i, lo), (i, hi)] for i, lo, hi in zip(x, l, h)], colors=colors, linewidths=2.0, zorder=3))
    w = 0.7
    bodies = [[(i - w / 2, min(a, b)), (i + w / 2, min(a, b)), (i + w / 2, max(a, b)), (i - w / 2, max(a, b))]
              for i, a, b in zip(x, o, c)]
    ax.add_collection(PolyCollection(bodies, facecolors=colors, edgecolors=colors, linewidths=0.8, zorder=4))
    for k, (p, m) in enumerate(mas):
        ax.plot(x, m[s], color=MA_COLORS[k % 3], lw=2.4, zorder=5, label=f"MA{p}")
    lo_p, hi_p = float(l.min()), float(h.max())
    pad = (hi_p - lo_p) * 0.06 or hi_p * 0.01
    ax.set_ylim(lo_p - pad - (hi_p - lo_p) * 0.14, hi_p + pad)   # bottom 14% reserved for volume bars
    ax.set_xlim(-1, show_n + 9)                                  # empty band on the right for the last-price tag
    vmax = float(v.max()) or 1.0
    vb = ax.get_ylim()
    vh = (vb[1] - vb[0]) * 0.12
    ax.bar(x, v / vmax * vh, bottom=vb[0], width=0.8, color=np.where(up, "#a5d6a7", "#ef9a9a"), zorder=1)
    last = float(c[-1])
    ax.axhline(last, color="#444444", lw=1.2, ls="--", zorder=2)
    ax.text(0.995, last, f"{last:.6g}", transform=ax.get_yaxis_transform(), va="bottom", ha="right", fontsize=13,
            color="white", bbox=dict(boxstyle="square,pad=0.2", fc="#444444", ec="none"), zorder=6)   # inside the plot: no overlap with axis labels
    ax.legend(loc="upper left", fontsize=13, frameon=True, framealpha=0.9, ncol=len(mas))
    ax.set_title(f"{sym} · {tf} · last {show_n} closed candles · MA{'/'.join(str(p) for p in MA_PERIODS)} · RSI{RSI_LEN} "
                 f"· last candle open {_fmt(int(t[-1]))} UTC", fontsize=15, loc="left", pad=8, fontweight="bold")
    plt_labels = [int(i) for i in np.linspace(4, show_n - 4, 8)]        # keep first/last labels inside the image
    ax.set_xticks(plt_labels)
    ax.tick_params(labelbottom=False)

    axr.plot(x, rs, color="#6a1b9a", lw=2.2, zorder=3)
    for lvl, st in ((70, "--"), (50, ":"), (30, "--")):
        axr.axhline(lvl, color="#888888", lw=1.1, ls=st)
    axr.set_ylim(0, 100)
    axr.set_yticks([30, 50, 70])
    axr.set_xticks(plt_labels)
    axr.set_xticklabels([_fmt(int(t[i])) for i in plt_labels], fontsize=12)
    axr.text(0.005, 0.86, f"RSI{RSI_LEN}", transform=axr.transAxes, fontsize=13, color="#6a1b9a", fontweight="bold")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, pil_kwargs={"compress_level": 3})
    return buf.getvalue()


def job_from_frame(symbol: str, tf: str, df: pd.DataFrame, show_n: int = SHOW_CANDLES) -> Dict[str, Any]:
    return {"symbol": symbol, "tf": tf, "show_n": show_n, "t": df["t"].to_numpy(np.int64), "o": df["open"].to_numpy(float),
            "h": df["high"].to_numpy(float), "l": df["low"].to_numpy(float), "c": df["close"].to_numpy(float),
            "v": df["volume"].to_numpy(float)}


# ───────────────────────────── batch + cache ─────────────────────────────
_CACHE: Dict[Tuple[str, str, int, int], bytes] = {}
_POOL: Optional[ProcessPoolExecutor] = None


def _safe_render(job: Dict[str, Any]) -> Tuple[Optional[bytes], str]:
    try:
        return render_chart(job), ""
    except Exception as exc:                                   # one bad chart must never kill the scan
        return None, f"{type(exc).__name__}: {exc}"


def render_many(jobs: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Tuple[Optional[bytes], str]]:
    """Render (with cache). Parallel on CHART_WORKERS processes; falls back to serial if the pool is unavailable.
    Returns {(symbol, tf): (png | None, error)}."""
    global _POOL
    out: Dict[Tuple[str, str], Tuple[Optional[bytes], str]] = {}
    todo = []
    for j in jobs:
        key = (j["symbol"], j["tf"], int(j["t"][-1]), j["show_n"])
        if key in _CACHE:
            out[(j["symbol"], j["tf"])] = (_CACHE[key], "")
        else:
            todo.append((key, j))
    results: List[Tuple[Optional[bytes], str]] = []
    if todo and CHART_WORKERS > 1:
        try:
            if _POOL is None:
                _POOL = ProcessPoolExecutor(max_workers=CHART_WORKERS, mp_context=multiprocessing.get_context("spawn"))   # spawn: safe next to asyncio/threads
            results = list(_POOL.map(_safe_render, [j for _, j in todo], chunksize=4))
        except (BrokenProcessPool, OSError, RuntimeError) as exc:
            logger.warning("Chart process pool failed (%s) → rendering serially", exc)
            _POOL = None
            results = []
    if todo and not results:
        results = [_safe_render(j) for _, j in todo]
    for (key, j), (png, err) in zip(todo, results):
        out[(j["symbol"], j["tf"])] = (png, err)
        if png:
            _CACHE[key] = png
    if len(_CACHE) > 1200:                                     # bound memory: drop the oldest half
        for k in list(_CACHE)[:600]:
            _CACHE.pop(k, None)
    return out
