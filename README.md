# V11 — Support/Resistance chart agent (4h · 1h · 15m · 5m)

Every 15 min, 12:00–19:00 Asia/Colombo (/on /off): top-60 liquid USDT perps → closed candles → **4 charts per coin**
(100 candles + MA20/MA50 + RSI14 + volume, 1536×1024) → 4 coins per API call (group i → key i, launches 1/sec) =
**OHLCV text of exactly the charted candles + 16 chart images** → `nvidia/nemotron-3.5-super-vl:free` reads S/R + patterns
from the charts and takes every price from the text → each call returns ≤1 best setup → a judge call picks the single best →
Telegram **admin chat only** → simulated tracking (/trades /pnl).

Script-side filtering: none for strategy (only output-format sanity; coins with a live trade or signalled in the last
DEDUPE_HOURS are skipped). Failures (chart/model/Binance) are logged and the admin gets a rate-limited alert.

## Run
    pip install -r requirements.txt
    cp .env.example .env     # fill TELEGRAM_* and OPENROUTER_API_KEY_1..15
    python3 cleanup_v10.py   # once, if old files are in the folder (deletes obsolete files, then itself)
    python3 main.py
Tests: `pip install pytest && python3 -m pytest tests -q`
