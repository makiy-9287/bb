"""
Configuration — DAY MODE with extended SMC/liquidity data.

Trading window: 13:00–20:00 SLST (London open to NY morning).
Trade monitor runs 24/7 regardless of window.
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
# "medium" maps to "high" internally per DeepSeek docs — use "high".
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
TOP_COINS_LIMIT      = 50
CANDLE_LIMIT         = 400           # upgraded 300 → 400
OHLCV_FETCH_LIMIT    = 499           # fetch more than limit for buffer

# ── Timeframes ───────────────────────────────────────────────────
TIMEFRAMES = ["4h", "1h", "15m"]

# ── Scan Interval ────────────────────────────────────────────────
SCAN_INTERVAL_MIN = 15

# ── SMC / Liquidity Tuning ───────────────────────────────────────
SWING_LENGTH        = 20
LIQUIDITY_RANGE_PCT = 0.01
POC_BINS            = 50

# Sweep detection
SWEEP_LOOKBACK      = 30     # scan last N candles for sweeps
SWEEP_PEN_ATR_MIN   = 0.15   # wick penetration must exceed this × ATR
EQUAL_LEVEL_TOL     = 0.001  # 0.1% tolerance for EQH/EQL clustering

# Displacement detection
DISPLACEMENT_ATR_MULT = 1.5  # body > 1.5 × ATR = displacement

# ── Execution Window (Sri Lanka Time, UTC+5:30) ──────────────────
# 13:00 SLST = 07:30 UTC = London open
# 20:00 SLST = 14:30 UTC = NY morning
SLST_START      = dt_time(13, 0)
SLST_END        = dt_time(20, 0)
SLST_UTC_OFFSET = 5.5

# ── Trading Days ─────────────────────────────────────────────────
TRADING_WEEKDAYS = {0, 1, 2, 3}
OFF_WEEKDAYS     = {4, 5, 6}
DAY_NAMES        = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# ── Trade Monitoring ─────────────────────────────────────────────
MONITOR_INTERVAL_SEC = 15
ACTIVE_TRADES_FILE   = "active_trades.json"
CLOSED_TRADES_FILE   = "closed_trades.json"

# ── Paths ────────────────────────────────────────────────────────
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
