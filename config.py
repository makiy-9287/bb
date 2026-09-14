"""
Configuration module. Loads all environment variables and defines
global constants used across the system.
"""
import os
from dotenv import load_dotenv
from datetime import time as dt_time

load_dotenv()

# ── API Credentials ──────────────────────────────────────────────
DEEPSEEK_API_KEY   = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL  = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL     = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")   # FIX: was deepseek-chat

# ── Reasoning (Thinking Mode) ────────────────────────────────────
# "medium" is requested literally; DeepSeek maps medium → high internally.
REASONING_EFFORT   = os.getenv("REASONING_EFFORT", "medium")
USE_THINKING       = os.getenv("USE_THINKING", "true").lower() == "true"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Binance ──────────────────────────────────────────────────────
BINANCE_FUTURES_REST = "https://fapi.binance.com"
TOP_COINS_LIMIT      = 70
CANDLE_LIMIT         = 400
OHLCV_FETCH_LIMIT    = 499      # CCXT-optimised (weight=2 vs 10)
TIMEFRAMES           = ["1d", "4h", "1h"]

# ── Strategy / SMC ───────────────────────────────────────────────
SWING_LENGTH       = 20
LIQUIDITY_RANGE_PCT = 0.01
POC_BINS           = 50

# ── Execution Window (Sri Lanka Time, UTC+5:30) ──────────────────
SLST_START       = dt_time(5, 0)
SLST_END         = dt_time(21, 0)
SLST_UTC_OFFSET  = 5.5

# ── Trade Monitoring ─────────────────────────────────────────────
MONITOR_INTERVAL_SEC = 15
ACTIVE_TRADES_FILE   = "active_trades.json"

# ── Paths ────────────────────────────────────────────────────────
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)