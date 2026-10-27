"""
V10 — Support/Resistance agent. Configuration.
OpenRouter (up to 15 keys) · 60 coins · 4h/1h/15m · 4 coins per API call · admin-only Telegram.
"""
import os
from datetime import time as dt_time

from dotenv import load_dotenv

load_dotenv()

# ── OpenRouter (up to 15 keys; group i → key i) ──────────────────
OPENROUTER_API_KEYS = [k for k in (os.getenv(f"OPENROUTER_API_KEY_{i}", "").strip() for i in range(1, 16)) if k]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL    = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-super-vl:free")
REASONING_EFFORT    = os.getenv("REASONING_EFFORT", "high")

# ── Telegram: ADMIN ONLY (signals, trade updates, commands) ───────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_ADMIN_ID  = os.getenv("TELEGRAM_ADMIN_ID", "")

# ── Binance Futures universe ─────────────────────────────────────
BINANCE_FUTURES_REST = "https://fapi.binance.com"
TOP_COINS_LIMIT      = 60
BLACKLIST_SYMBOLS = {"USDCUSDT", "BUSDUSDT", "TUSDUSDT", "FDUSDUSDT", "DAIUSDT", "USDPUSDT", "EURUSDT"}
MIN_QUOTE_VOLUME_USDT = float(os.getenv("MIN_QUOTE_VOLUME_USDT", "30000000"))   # 24h, data-quality / liquidity floor
MAX_SPREAD_PCT        = 0.05
MAX_DATA_STALE_BARS   = 3

# ── Timeframes: 4h 200 · 1h 200 · 15m 150 (closed candles only) ───
TIMEFRAMES    = ["4h", "1h", "15m", "5m"]
SHOW_CANDLES  = 100                                  # candles drawn in each chart AND written to the text payload
WARMUP_CANDLES = 150                                 # extra history so MA/RSI are accurate on the first drawn candle
CANDLE_LIMITS = {tf: SHOW_CANDLES + WARMUP_CANDLES for tf in TIMEFRAMES}
COINS_PER_CALL = 4                                   # 60 coins → 15 calls → 15 keys (4 coins × 4 charts = 16 images / call)

# ── Charts (vision input): 1536×1024 = 3×2 tiles of 512 px (native tile size of the Nemotron VL encoder) ──
CHART_W, CHART_H = 1536, 1024
MA_PERIODS = (20, 50)
RSI_LEN    = 14
CHART_WORKERS = int(os.getenv("CHART_WORKERS", str(min(8, os.cpu_count() or 1))))
ALERT_COOLDOWN_MIN = 30                              # admin alerts about failures are rate-limited

# ── Scan schedule: every 15 minutes, 12:00–19:00 Asia/Colombo ─────
SCAN_INTERVAL_MIN = 15
SCAN_OFFSET_SEC   = 10                               # wait after the candle close so exchange data settles
TIMEZONE      = "Asia/Colombo"                       # UTC+05:30, no DST
WINDOW_START  = dt_time(12, 0)                       # inclusive
WINDOW_END    = dt_time(19, 0)                       # exclusive → last scan 18:45
_DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
TRADING_WEEKDAYS = {i for i, d in enumerate(_DAYS)
                    if d in {x.strip().lower()[:3] for x in os.getenv("SCAN_DAYS", ",".join(_DAYS)).split(",")}} or set(range(7))
START_ENABLED = os.getenv("START_ENABLED", "1") == "1"

# ── Repeat protection (NOT a quality filter) ─────────────────────
DEDUPE_HOURS = float(os.getenv("DEDUPE_HOURS", "6"))   # a coin that was just signalled is not re-sent / re-analysed

# ── Trade tracking (simulated) ───────────────────────────────────
MONITOR_INTERVAL_SEC = 15
ACTIVE_TRADES_FILE   = "active_trades.json"
CLOSED_TRADES_FILE   = "closed_trades.json"
SIGNAL_EXPIRY_MIN    = 240                           # unfilled LIMIT expires
MAX_HOLD_HOURS       = 48
TP_SPLIT             = (0.4, 0.3, 0.3)
MOVE_SL_TO_BE_AFTER_TP1 = False

# ── Cost model (used for net R only; never to filter signals) ────
FEE_MAKER    = 0.0002
FEE_TAKER    = 0.0005
SLIPPAGE_BPS = 2.0
FUNDING_8H   = 0.0001
ASSUMED_HOLD_HOURS = 8

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
