"""
v4.1 Configuration — Single-request mode (no batching).
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
TOP_COINS_LIMIT      = 40

# ── Candle counts per timeframe ──────────────────────────────────
CANDLES_4H    = 150
CANDLES_1H    = 100
CANDLES_15M   = 100
MAX_CANDLES   = 359

TIMEFRAMES = ["4h", "1h", "15m"]

CANDLE_COUNTS = {
    "4h":  CANDLES_4H,
    "1h":  CANDLES_1H,
    "15m": CANDLES_15M,
}

# ── Scan ─────────────────────────────────────────────────────────
SCAN_INTERVAL_MIN = 15
# NOTE: No batching in v4.1 — all coins sent in ONE request.

# ── Trade Management ─────────────────────────────────────────────
MAX_ACTIVE_TRADES    = 10
BATCH_SIZE        = 10
MONITOR_INTERVAL_SEC = 5
ACTIVE_TRADES_FILE   = "active_trades.json"
CLOSED_TRADES_FILE   = "closed_trades.json"

# ── Execution Window (Sri Lanka Time, UTC+5:30) ──────────────────
SLST_START       = dt_time(8, 0)
SLST_END         = dt_time(20, 0)
SLST_UTC_OFFSET  = 5.5

# ── Trading Days ─────────────────────────────────────────────────
TRADING_WEEKDAYS = {1, 2, 3}
OFF_WEEKDAYS     = {4, 5, 6}
DAY_NAMES        = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# ── Paths ────────────────────────────────────────────────────────
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
