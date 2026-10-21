"""
v5 Configuration — OpenRouter Multi-Key + 1D Timeframe.
"""
import os
from dotenv import load_dotenv
from datetime import time as dt_time

load_dotenv()

# ── OpenRouter Credentials ───────────────────────────────────────
# Keys list එකක් විදිහට ගන්නවා (comma separated)
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
TOP_COINS_LIMIT      = 70   # 70 Coins
BATCH_SIZE           = 10   # 10 ගානෙ Batch යනවා (7 batches per scan)

# ── Timeframes & Candles (1D එකතු කළා) ───────────────────────────
TIMEFRAMES = ["1d", "4h", "1h", "15m"]

CANDLES_1D  = 50    # දවස් 50 (Macro Trend)
CANDLES_4H  = 150
CANDLES_1H  = 100
CANDLES_15M = 100
MAX_CANDLES = 499

CANDLE_COUNTS = {
    "1d":  CANDLES_1D,
    "4h":  CANDLES_4H,
    "1h":  CANDLES_1H,
    "15m": CANDLES_15M,
}

# ── Scan Settings ────────────────────────────────────────────────
SCAN_INTERVAL_MIN = 30

# ── Trade Management ─────────────────────────────────────────────
MAX_ACTIVE_TRADES    = 50
MONITOR_INTERVAL_SEC = 5
ACTIVE_TRADES_FILE   = "active_trades.json"
CLOSED_TRADES_FILE   = "closed_trades.json"

# ── Execution Window (10:00 AM - 08:00 PM SLST) ──────────────────
SLST_START       = dt_time(9, 0)
SLST_END         = dt_time(20, 0)
SLST_UTC_OFFSET  = 5.5

# ── Trading Days ─────────────────────────────────────────────────
TRADING_WEEKDAYS = {0, 1, 2, 3, 4}
OFF_WEEKDAYS     = { 5, 6}
DAY_NAMES        = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
