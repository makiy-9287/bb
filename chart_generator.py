"""
Chart generator — produces 1920×1080 candlestick charts with
MA(7,25,99) + RSI + Volume, using mplfinance.

Multiprocessing-aware: safe to call from asyncio via run_in_executor.
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
    CHART_DPI, CHART_WORKERS, CHART_CLEANUP,
)

logger = logging.getLogger(__name__)


# ── Worker function (must be top-level for pickling) ─────────────
def _generate_single_chart(args: Tuple) -> Optional[str]:
    """
    Worker: generate one chart. Runs in a separate process.
    args = (symbol, timeframe, df_dict, output_path)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplfinance as mpf
    import pandas_ta_classic as ta

    symbol, timeframe, df_dict, output_path = args

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

        # Indicators
        ma7  = ta.sma(df["Close"], length=7)
        ma25 = ta.sma(df["Close"], length=25)
        ma99 = ta.sma(df["Close"], length=99)
        rsi  = ta.rsi(df["Close"], length=14)

        apds = [
            mpf.make_addplot(ma7,  panel=0, color="#f0b90b", width=1.2),
            mpf.make_addplot(ma25, panel=0, color="#e91e63", width=1.2),
            mpf.make_addplot(ma99, panel=0, color="#9c27b0", width=1.2),
            mpf.make_addplot(rsi,  panel=2, color="#f0b90b", width=1.2,
                             ylabel="RSI", ylim=(0, 100)),
            mpf.make_addplot(pd.Series(70, index=df.index), panel=2,
                             color="#f6465d", linestyle="--", width=0.8),
            mpf.make_addplot(pd.Series(30, index=df.index), panel=2,
                             color="#0ecb81", linestyle="--", width=0.8),
        ]

        # Binance dark theme
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
            figsize=(CHART_WIDTH_PX / CHART_DPI, CHART_HEIGHT_PX / CHART_DPI),
            dpi=CHART_DPI,
            title=f"{symbol}  |  {timeframe.upper()}",
            tight_layout=True,
            savefig=dict(fname=output_path, dpi=CHART_DPI,
                         bbox_inches="tight", pad_inches=0.1),
        )
        plt.close("all")
        return output_path

    except Exception as exc:
        logger.warning("Chart %s %s failed: %s", symbol, timeframe, exc)
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass
        return None


# ── Public API ───────────────────────────────────────────────────
class ChartManager:
    """Manages chart generation, storage, and cleanup per cycle."""

    def __init__(self):
        self._current_dir: Optional[str] = None

    def new_cycle_dir(self) -> str:
        """Create a fresh temp folder for the current cycle."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(CHARTS_DIR, ts)
        os.makedirs(path, exist_ok=True)
        self._current_dir = path
        return path

    def cleanup(self, cycle_dir: Optional[str] = None):
        """Delete charts after agent responds."""
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
        """Remove ALL old chart folders on startup."""
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


async def generate_all_charts(
    bundle: Dict[str, Dict[str, pd.DataFrame]],
    symbols: List[str],
    cycle_dir: str,
) -> Dict[str, List[str]]:
    """
    Generate all charts for the given symbols using multiprocessing.

    Returns: {symbol: [chart_path_4h, chart_path_1h, chart_path_15m]}
    """
    import asyncio

    tasks_specs = []
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
                rec["ts"] = rec["ts"].isoformat() if hasattr(rec["ts"], "isoformat") else rec["ts"]
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

    # Verify files exist
    for sym, paths in output_map.items():
        output_map[sym] = [p for p in paths if os.path.exists(p)]

    return output_map


chart_manager = ChartManager()
