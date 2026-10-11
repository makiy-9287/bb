"""
Configuration — DAY MODE (Mon–Thu, 12:00–21:00 SLST).
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

# Admin chat — command authorization + always receives signals
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Broadcast list — comma-separated chat IDs that receive signals.
# Admin is automatically prepended (no duplicates).
_raw_broadcast = os.getenv("TELEGRAM_BROADCAST_IDS", "")
_broadcast_list = [x.strip() for x in _raw_broadcast.split(",") if x.strip()]
if TELEGRAM_CHAT_ID and TELEGRAM_CHAT_ID not in _broadcast_list:
    _broadcast_list.insert(0, TELEGRAM_CHAT_ID)
TELEGRAM_BROADCAST_IDS = _broadcast_list

# ── Binance ──────────────────────────────────────────────────────
BINANCE_FUTURES_REST = "https://fapi.binance.com"
TOP_COINS_LIMIT      = 50
CANDLE_LIMIT         = 300
OHLCV_FETCH_LIMIT    = 399

TIMEFRAMES = ["4h", "1h", "15m"]
SCAN_INTERVAL_MIN = 15

SWING_LENGTH        = 20
LIQUIDITY_RANGE_PCT = 0.01
POC_BINS            = 50

# ── Execution Window (Sri Lanka Time, UTC+5:30) ──────────────────
SLST_START       = dt_time(12, 0)
SLST_END         = dt_time(21, 0)
SLST_UTC_OFFSET  = 5.5

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
