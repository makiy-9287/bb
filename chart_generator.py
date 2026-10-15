"""
Chart generator — 1920×1080 PNG with MA(7,25,99) + RSI + Volume.
Optimized PNG compression + file size validation.
"""
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import (
    CHARTS_DIR, CHART_WIDTH_PX, CHART_HEIGHT_PX,
    CHART_DPI, CHART_WORKERS, CHART_MAX_FILE_KB,
)

logger = logging.getLogger(__name__)


# ── Worker (top-level for pickling) ──────────────────────────────
def _generate_single_chart(args: Tuple) -> Optional[str]:
    """Worker: generate one chart in an isolated process."""
    symbol, timeframe, df_dict, output_path = args

    # Local imports (required inside worker for ProcessPoolExecutor)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplfinance as mpf
    import pandas_ta_classic as ta

    try:
        df = pd.DataFrame(df_dict)
        if df.empty or len(df) < 30:
            return None

        df["ts"] = pd.to_datetime(df["ts"])
        df = df.set_index("ts")
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df = df.tail(300)

        ma7  = ta.sma(df["Close"], length=7)
        ma25 = ta.sma(df["Close"], length=25)
        ma99 = ta.sma(df["Close"], length=99)
        rsi  = ta.rsi(df["Close"], length=14)

        # Horizontal RSI guide lines
        rsi_70 = pd.Series(70, index=df.index)
        rsi_30 = pd.Series(30, index=df.index)

        apds = [
            mpf.make_addplot(ma7,  panel=0, color="#f0b90b", width=1.2),
            mpf.make_addplot(ma25, panel=0, color="#e91e63", width=1.2),
            mpf.make_addplot(ma99, panel=0, color="#9c27b0", width=1.2),
            mpf.make_addplot(rsi,  panel=2, color="#f0b90b", width=1.2,
                             ylabel="RSI", ylim=(0, 100)),
            mpf.make_addplot(rsi_70, panel=2, color="#f6465d",
                             linestyle="--", width=0.8),
            mpf.make_addplot(rsi_30, panel=2, color="#0ecb81",
                             linestyle="--", width=0.8),
        ]

        mc = mpf.make_marketcolors(
            up="#0ecb81", down="#f6465d",
            edge={"up": "#0ecb81", "down": "#f6465d"},
            wick={"up": "#0ecb81", "down": "#f6465d"},
            volume={"up": "#0ecb81", "down": "#f6465d"},
        )
        style = mpf.make_mpf_style(
            marketcolors=mc,
            facecolor="#0b0e11",
            edgecolor="#2b3139",
            figcolor="#0b0e11",
            gridcolor="#1e2329",
            gridstyle="--",
            y_on_right=True,
            rc={
                "axes.labelcolor": "#848e9c",
                "xtick.color": "#848e9c",
                "ytick.color": "#848e9c",
                "font.size": 11,
                "axes.titlesize": 14,
            },
        )

        mpf.plot(
            df,
            type="candle",
            style=style,
            volume=True,
            addplot=apds,
            panel_ratios=(6, 2, 2),
            figsize=(CHART_WIDTH_PX / CHART_DPI,
                     CHART_HEIGHT_PX / CHART_DPI),
            dpi=CHART_DPI,
            title=f"{symbol}  |  {timeframe.upper()}",
            tight_layout=True,
            savefig=dict(fname=output_path, dpi=CHART_DPI,
                         bbox_inches="tight", pad_inches=0.1,
                         optimize=True),
        )
        plt.close("all")

        # Validate file size
        if not os.path.exists(output_path):
            return None
        size_kb = os.path.getsize(output_path) / 1024
        if size_kb > CHART_MAX_FILE_KB:
            logger.warning("Chart too big (%.0f KB) — rejecting %s %s",
                           size_kb, symbol, timeframe)
            try:
                os.remove(output_path)
            except Exception:
                pass
            return None
        return output_path

    except Exception as exc:
        logger.warning("Chart %s %s failed: %s", symbol, timeframe, exc)
        try:
            plt.close("all")
        except Exception:
            pass
        return None


# ── Manager ──────────────────────────────────────────────────────
class ChartManager:
    def __init__(self):
        self._current_dir: Optional[str] = None

    def new_cycle_dir(self, tag: str = "") -> str:
        """Create a fresh temp folder. Tag = short label for logging."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        name = f"{ts}_{tag}" if tag else ts
        path = os.path.join(CHARTS_DIR, name)
        os.makedirs(path, exist_ok=True)
        self._current_dir = path
        return path

    def cleanup(self, cycle_dir: Optional[str] = None):
        target = cycle_dir or self._current_dir
        if not target or not os.path.isdir(target):
            return
        try:
            shutil.rmtree(target)
            logger.info("Charts cleaned: %s", target)
        except Exception as exc:
            logger.warning("Chart cleanup failed %s: %s", target, exc)
        finally:
            if target == self._current_dir:
                self._current_dir = None

    def startup_cleanup(self):
        try:
            if not os.path.isdir(CHARTS_DIR):
                return
            for entry in os.listdir(CHARTS_DIR):
                full = os.path.join(CHARTS_DIR, entry)
                if os.path.isdir(full):
                    shutil.rmtree(full, ignore_errors=True)
            logger.info("Startup chart cleanup complete")
        except Exception as exc:
            logger.warning("Startup cleanup failed: %s", exc)


chart_manager = ChartManager()


async def generate_all_charts(
    bundle: Dict[str, Dict[str, pd.DataFrame]],
    symbols: List[str],
    cycle_dir: str,
) -> Dict[str, List[str]]:
    import asyncio

    tasks_specs: List[Tuple] = []
    output_map: Dict[str, List[str]] = {}

    for symbol in symbols:
        if symbol not in bundle:
            continue
        tf_data = bundle[symbol]
        output_map[symbol] = []
        for tf in ("4h", "1h", "15m"):
            df = tf_data.get(tf)
            if df is None or df.empty:
                continue
            out_path = os.path.join(cycle_dir, f"{symbol}_{tf}.png")
            df_records = df.to_dict(orient="records")
            for rec in df_records:
                ts = rec.get("ts")
                rec["ts"] = ts.isoformat() if hasattr(ts, "isoformat") else ts
            tasks_specs.append((symbol, tf, df_records, out_path))
            output_map[symbol].append(out_path)

    if not tasks_specs:
        return output_map

    loop = asyncio.get_event_loop()
    logger.info("Generating %d charts (workers=%d) …",
                len(tasks_specs), CHART_WORKERS)

    try:
        with ProcessPoolExecutor(max_workers=CHART_WORKERS) as executor:
            futures = [
                loop.run_in_executor(executor, _generate_single_chart, spec)
                for spec in tasks_specs
            ]
            results = await asyncio.gather(*futures, return_exceptions=True)
    except Exception as exc:
        logger.error("Chart generation pool failed: %s", exc)
        return output_map

    success = sum(1 for r in results
                  if not isinstance(r, Exception) and r is not None)
    logger.info("Charts generated: %d/%d", success, len(tasks_specs))

    # Drop failed paths from output_map
    for sym in list(output_map.keys()):
        output_map[sym] = [p for p in output_map[sym] if os.path.exists(p)]

    return output_map
