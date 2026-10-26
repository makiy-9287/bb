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
TIMEFRAMES    = ["1h", "15m", "5m"]        # scalp: 1h bias · 15m setup/liquidity · 5m entry
CANDLE_LIMITS = {"1h": 250, "15m": 200, "5m": 100, "3m": 100}
OHLCV_FETCH_LIMIT = 300

# ── Scan ─────────────────────────────────────────────────────────
SCAN_INTERVAL_MIN = 5            # a scan after every 5m candle close
SCAN_OFFSET_SEC   = 8            # wait this long after the close (exchange data settles)

# ── Strategy / SMC ───────────────────────────────────────────────
SWING_LENGTH        = 20
LIQUIDITY_RANGE_PCT = 0.01
POC_BINS            = 50
EQ_LEVEL_TOLERANCE  = 0.001      # 0.1% for equal highs/lows

# ── Execution window: Tue–Thu, 12:00 (incl) – 21:00 (excl), Asia/Colombo ──
TIMEZONE         = "Asia/Colombo"        # UTC+05:30, no DST
TRADING_WEEKDAYS = {1, 2, 3}             # Mon=0 … Sun=6  → Tue, Wed, Thu
WINDOW_START     = dt_time(12, 0)
WINDOW_END       = dt_time(21, 0)
# legacy names kept so older imports keep working
SLST_START, SLST_END, SLST_UTC_OFFSET = WINDOW_START, WINDOW_END, 5.5
OFF_WEEKDAYS = set(range(7)) - TRADING_WEEKDAYS
DAY_NAMES    = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# ── Trade monitoring ─────────────────────────────────────────────
MONITOR_INTERVAL_SEC = 15
ACTIVE_TRADES_FILE   = "active_trades.json"
CLOSED_TRADES_FILE   = "closed_trades.json"

# ── Paths ────────────────────────────────────────────────────────
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ── Universe / liquidity ─────────────────────────────────────────
MIN_QUOTE_VOLUME_USDT = float(os.getenv("MIN_QUOTE_VOLUME_USDT", "30000000"))  # 24h
MAX_SPREAD_PCT        = 0.05             # best bid/ask spread, % of mid
VOLUME_SPIKE_REJECT   = 8.0              # last-24h volume vs prior daily average
MAX_DATA_STALE_BARS   = 3                # last closed candle older than N bars → stale

# ── ICT engine ───────────────────────────────────────────────────
SWING_LR            = {"1h": 3, "15m": 3, "5m": 3}   # candles each side   # candles each side
ATR_LEN             = 14
EQ_TOL_ATR          = 0.15               # equal highs/lows tolerance (× ATR)
SWEEP_MIN_PEN_ATR   = 0.10               # wick must pierce level by ≥ 0.10 ATR …
SWEEP_MAX_PEN_ATR   = 2.50               # … but ≤ 2.5 ATR (deeper = real breakout)
SWEEP_RECLAIM_BARS  = 3                  # close back through level within N candles
SWEEP_LOOKBACK_BARS = 8                 # sweep must be this recent (15m bars)
MSS_MAX_AGE_BARS = 8                 # MSS candle must be this recent (15m bars)
DISP_BODY_ATR       = 0.80               # displacement: body ≥ 0.8 ATR …
DISP_BODY_RANGE     = 0.60               # … and body ≥ 60% of the candle range
SL_ATR_BUFFER       = 0.5                # SL = invalidation ∓ 0.5 × ATR(15m)
SL_MIN_ATR          = 0.8
MAX_ENTRY_DIST_PCT = 1.0
REQUIRE_5M          = True               # strict: no 5m refinement → reject
SHORTLIST_MAX       = 12

# ── Costs, RR, risk (all configurable) ───────────────────────────
MIN_NET_RR          = float(os.getenv("MIN_NET_RR", "1.5"))   # net of fees/slippage/funding, to TP1
MAX_SL_PCT          = float(os.getenv("MAX_SL_PCT", "1.5"))   # reject (not clamp) above this
MIN_RANK_SCORE      = 60                 # heuristic quality floor — NOT a probability
FEE_MAKER           = 0.0002             # 0.02 %
FEE_TAKER           = 0.0005             # 0.05 %
SLIPPAGE_BPS        = 2.0                # on market entries / stop exits
FUNDING_8H          = 0.0001             # assumed funding per 8h (longs pay if +)
ASSUMED_HOLD_HOURS = 2
TP_SPLIT            = (0.4, 0.3, 0.3)    # partial exit fractions at TP1/TP2/TP3
MOVE_SL_TO_BE_AFTER_TP1 = False
SIGNAL_EXPIRY_MIN = 30                # unfilled LIMIT expires
COOLDOWN_MIN_AFTER_CLOSE = 90           # same symbol blocked after a closed trade
MAX_OPEN_TRADES     = 3
MAX_SAME_DIRECTION  = 2                  # alts are highly correlated
ACCOUNT_RISK_PCT    = float(os.getenv("ACCOUNT_RISK_PCT", "0") or 0)  # 0 = not supplied → no sizing
ENGINE_ONLY_MODE    = os.getenv("ENGINE_ONLY_MODE", "0") == "1"  # skip LLM (no keys / testing)
TOOL_STRICT_SCHEMA  = os.getenv("TOOL_STRICT_SCHEMA", "0") == "1"
MAX_ENTRY_DIST_ATR = 2.0                # LIMIT entry max distance from price (× ATR 15m)
MAX_TARGET_PCT = 5.0               # ignore structural targets farther than this
MAX_HOLD_HOURS = 4                 # open trade closed "TIME" at last close after this

# ── Scalp / hybrid runtime ───────────────────────────────────────
USE_SESSION_WINDOW   = os.getenv("USE_SESSION_WINDOW", "0") == "1"   # default: NO time limit (manual /on /off)
START_ENABLED        = os.getenv("START_ENABLED", "1") == "1"
ALLOW_REVERSAL       = os.getenv("ALLOW_REVERSAL", "0") == "1"      # default: trade WITH the 1h trend only
BTC_FILTER           = os.getenv("BTC_FILTER", "1") == "1"
MAX_AGENT_BATCH      = 6        # best N confirmed setups of one scan go to ONE api key
SETUP_COOLDOWN_MIN   = 60       # same setup fingerprint is never sent to the model twice
SIGNAL_MAX_AGE_SEC   = 600      # model answer older than this (or price run away) → dropped
KEY_FAILOVER_TRIES   = 3        # if the assigned key fails, try the next keys (bounded)

# ── Manual AI payload (manual_payload.py) ────────────────────────
MANUAL_TOP_N   = 30
MANUAL_COUNTS  = {"1h": 48, "15m": 64, "5m": 60, "3m": 40}     # candles per TF written to the file
MANUAL_MIN_ATR15_PCT = 0.10                                    # skip dead coins (15m ATR % of price)
MANUAL_MAX_TOKENS_PER_FILE = 90_000                           # rough estimate; above this the file is split in parts
