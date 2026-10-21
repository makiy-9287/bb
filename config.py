"""
v6 Configuration — Parallel Multi-Key, 60 Coins, 15-Point Scoring.
"""
import os
from dotenv import load_dotenv
from datetime import time as dt_time

load_dotenv()

# ── OpenRouter Credentials ───────────────────────────────────────
_raw_keys = os.getenv("OPENROUTER_API_KEYS", "")
OPENROUTER_API_KEYS = [k.strip() for k in _raw_keys.split(",") if k.strip()]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL    = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")

# ── Reasoning ────────────────────────────────────────────────────
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "high")

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
TOP_COINS_LIMIT      = 60   # 60 Coins
BATCH_SIZE           = 4    # 4 Coins per API call (60/4 = 15 batches)
MAX_CANDLES          = 300  # Fetch buffer

# ── Timeframes & Candles ─────────────────────────────────────────
TIMEFRAMES = ["1d", "4h", "1h", "15m"]

CANDLES_1D  = 100
CANDLES_4H  = 250
CANDLES_1H  = 200
CANDLES_15M = 150

CANDLE_COUNTS = {
    "1d":  CANDLES_1D,
    "4h":  CANDLES_4H,
    "1h":  CANDLES_1H,
    "15m": CANDLES_15M,
}

# ── Scan Settings ────────────────────────────────────────────────
SCAN_INTERVAL_MIN = 15

# ── Trade Management ─────────────────────────────────────────────
MAX_ACTIVE_TRADES    = 10
MONITOR_INTERVAL_SEC = 5
ACTIVE_TRADES_FILE   = "active_trades.json"
CLOSED_TRADES_FILE   = "closed_trades.json"

# ── Execution Window (09:30 AM - 05:00 PM SLST) ──────────────────
SLST_START       = dt_time(9, 30)
SLST_END         = dt_time(17, 0)
SLST_UTC_OFFSET  = 5.5

# ── Trading Days ─────────────────────────────────────────────────
TRADING_WEEKDAYS = {0, 1, 2, 3, 4}
OFF_WEEKDAYS     = {5, 6}
DAY_NAMES        = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
