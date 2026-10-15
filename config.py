"""
v3 Configuration — Hybrid vision + text system.
Trading window: 12:00–20:00 SLST (Mon–Thu).
"""
import os
from dotenv import load_dotenv
from datetime import time as dt_time

load_dotenv()

# ── API Credentials ──────────────────────────────────────────────
DEEPSEEK_API_KEY   = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL  = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL     = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")

# ── Reasoning ────────────────────────────────────────────────────
REASONING_EFFORT   = os.getenv("REASONING_EFFORT", "high")
USE_THINKING       = os.getenv("USE_THINKING", "true").lower() == "true"

# ── Telegram ─────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

_raw = os.getenv("TELEGRAM_BROADCAST_IDS", "")
_ids = [x.strip() for x in _raw.split(",") if x.strip()]
if TELEGRAM_CHAT_ID and TELEGRAM_CHAT_ID not in _ids:
    _ids.insert(0, TELEGRAM_CHAT_ID)
TELEGRAM_BROADCAST_IDS = _ids

# ── Binance ──────────────────────────────────────────────────────
BINANCE_FUTURES_REST = "https://fapi.binance.com"
TOP_COINS_LIMIT      = 40          # v3: 40 coins
CANDLE_LIMIT         = 300         # v3: 300 candles per TF
OHLCV_FETCH_LIMIT    = 499         # fetch more for EMA200 buffer

# ── Timeframes ───────────────────────────────────────────────────
TIMEFRAMES = ["4h", "1h", "15m"]

# ── Scan ─────────────────────────────────────────────────────────
SCAN_INTERVAL_MIN = 15
SCAN_WINDOW_START = 12            # 12:00 SLST
SCAN_WINDOW_END   = 20            # 20:00 SLST

# ── Chart Generation ─────────────────────────────────────────────
CHART_COIN_LIMIT   = 40           # generate charts for these many coins
CHART_WORKERS      = 3            # parallel chart generation processes
CHART_WIDTH_PX     = 1920
CHART_HEIGHT_PX    = 1080
CHART_DPI          = 100
CHART_CLEANUP      = True         # auto-delete after each cycle
CHART_RAW_CANDLES  = 20           # raw candles in text payload for SL/TP
CHART_TIMEOUT_SEC  = 180          # max time for all chart generation

# ── SMC / Liquidity Tuning ───────────────────────────────────────
SWING_LENGTH        = 20
LIQUIDITY_RANGE_PCT = 0.01
POC_BINS            = 50
SWEEP_LOOKBACK      = 30
SWEEP_PEN_ATR_MIN   = 0.15
EQUAL_LEVEL_TOL     = 0.001
DISPLACEMENT_ATR_MULT = 1.5

# ── Trade Management ─────────────────────────────────────────────
MAX_ACTIVE_TRADES    = 3          # max concurrent positions
MONITOR_INTERVAL_SEC = 5          # v3: 5s for fast SL
ACTIVE_TRADES_FILE   = "active_trades.json"
CLOSED_TRADES_FILE   = "closed_trades.json"

# ── Execution Window (Sri Lanka Time, UTC+5:30) ──────────────────
SLST_START       = dt_time(12, 0)
SLST_END         = dt_time(20, 0)
SLST_UTC_OFFSET  = 5.5

# ── Trading Days ─────────────────────────────────────────────────
TRADING_WEEKDAYS = {0, 1, 2, 3}
OFF_WEEKDAYS     = {4, 5, 6}
DAY_NAMES        = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# ── Paths ────────────────────────────────────────────────────────
DATA_DIR    = "data"
CHARTS_DIR  = os.path.join(DATA_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)
