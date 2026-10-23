"""
v6 ULTRA — Configuration.
OpenRouter + Nemotron 3 Ultra, 15 API keys, 60 coins, 4 timeframes.
"""
import os
from dotenv import load_dotenv
from datetime import time as dt_time

load_dotenv()

# ── OpenRouter (15 keys) ─────────────────────────────────────────
OPENROUTER_API_KEYS = []
for i in range(1, 16):
    k = os.getenv(f"OPENROUTER_API_KEY_{i}", "").strip()
    if k:
        OPENROUTER_API_KEYS.append(k)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL    = "nvidia/nemotron-3-ultra-550b-a55b:free"
REASONING_EFFORT    = os.getenv("REASONING_EFFORT", "high")

# ── Telegram ─────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_ADMIN_ID  = os.getenv("TELEGRAM_ADMIN_ID", "")
TELEGRAM_BROADCAST_IDS = [
    x.strip() for x in os.getenv("TELEGRAM_BROADCAST_IDS", "").split(",")
    if x.strip()
]

# ── Binance Futures ──────────────────────────────────────────────
BINANCE_FUTURES_REST = "https://fapi.binance.com"
TOP_COINS_LIMIT      = 60
# Exclude only stablecoin / wrapped pairs (not a real "blacklist")
BLACKLIST_SYMBOLS = {
    "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "FDUSDUSDT",
    "DAIUSDT", "USDPUSDT", "EURUSDT",
}

# ── Timeframes & candle limits ───────────────────────────────────
TIMEFRAMES    = ["1d", "4h", "1h", "15m"]
CANDLE_LIMITS = {"1d": 100, "4h": 200, "1h": 150, "15m": 100}
OHLCV_FETCH_LIMIT = 200          # weight=2, covers all TFs

# ── Scan ─────────────────────────────────────────────────────────
SCAN_INTERVAL_MIN = 15

# ── Strategy / SMC ───────────────────────────────────────────────
SWING_LENGTH        = 20
LIQUIDITY_RANGE_PCT = 0.01
POC_BINS            = 50
EQ_LEVEL_TOLERANCE  = 0.001      # 0.1% for equal highs/lows

# ── Execution window (Sri Lanka Time, UTC+5:30) ──────────────────
SLST_START      = dt_time(8, 0)
SLST_END        = dt_time(20, 0)
SLST_UTC_OFFSET = 5.5

# ── Trading days ─────────────────────────────────────────────────
OFF_WEEKDAYS = {0, 4, 5, 6}         # Fri, Sat, Sun
DAY_NAMES    = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# ── Trade monitoring ─────────────────────────────────────────────
MONITOR_INTERVAL_SEC = 15
ACTIVE_TRADES_FILE   = "active_trades.json"
CLOSED_TRADES_FILE   = "closed_trades.json"

# ── Paths ────────────────────────────────────────────────────────
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ── A+ signal filter (signal_filter.py) ──────────────────────────
MIN_RR              = 1.5    # RR measured to TP1 (real numbers, not the model's claim)
MAX_SL_PCT          = 2.0    # max SL distance from entry, %  (day-trade)
SL_ATR_BUFFER       = 1.5    # SL = 15m structure anchor ± 0.5 × ATR(15m)
SL_MIN_ATR          = 2.0   # SL at least 0.8 × ATR(15m) from entry
MAX_ENTRY_DIST_PCT  = 2.5    # LIMIT entry max distance from current price, %
MAX_TP1_PCT         = 4.0    # TP1 must be reachable today
MAX_TP3_PCT         = 10.0
MIN_SCORE           = 85   # 0-100 quality score; below this → signal is discarded
