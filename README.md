# V10 — Support/Resistance agent (4h · 1h · 15m)

Every 15 min, 12:00–19:00 Asia/Colombo (/on /off): top-60 liquid USDT perps → closed candles (4h×200, 1h×200, 15m×150)
→ 4 coins per API call (group i → key i, launches 1/sec) → **the prompt** finds high-probability S/R zones with 15m
confirmation, entry, SL and next-level targets; each call returns ≤1 best setup → a judge call picks the single best →
Telegram **admin chat only** → simulated tracking (/trades /pnl).

Script-side filtering: none for strategy. It only checks the model's output is well-formed (known symbol, finite numbers,
SL/TP on the right side) and skips coins that already have a live trade or were signalled in the last DEDUPE_HOURS.

## Run
    pip install -r requirements.txt
    cp .env.example .env     # fill TELEGRAM_* and OPENROUTER_API_KEY_1..15
    python3 cleanup_v10.py   # once, if old files are in the folder (deletes obsolete files, then itself)
    python3 main.py
Tests: `pip install pytest && python3 -m pytest tests -q`
