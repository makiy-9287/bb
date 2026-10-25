# Scalp Hybrid Agent (Binance USDT-M perps)

**Flow (every 5m candle close, Python only):** 60 liquid perps → closed-candle data (1h×250, 15m×200, 5m×100)
→ 1h bias · 15m liquidity sweep + displacement MSS + OB/FVG zone · 5m entry confirmation → engine-confirmed setups.
If a scan has setups they are sent as ONE payload to the NEXT OpenRouter key (1→15→1 round-robin, background call,
failover to following keys). The model picks ≤1; server re-validates every number; fresh-price gate; Telegram; tracking.

## Setup
    pip install -r requirements.txt
    cp .env.example .env      # fill TELEGRAM_* and OPENROUTER_API_KEY_1..15  (never commit .env)
    python main.py            # control from Telegram: /on /off /status /trades /pnl /scan
No LLM keys? `ENGINE_ONLY_MODE=1`. Tests: `pip install pytest && python -m pytest tests -q`.
Backtest (engine only, no LLM): `python backtest.py --csv BTCUSDT_5m.csv --rr 1.5 2.0` (or `--synthetic` smoke test).

## Files
main.py (scan loop, round-robin, gates) · ict_engine.py (causal ICT engine, definitions in its docstring) ·
data_fetcher.py/ohlcv.py (universe + closed-candle hygiene) · risk.py (net RR, limits) · agent.py (prompt, schema,
validation, queue) · trade_monitor.py/performance.py (state machine, net R stats) · backtest.py · schedule.py (optional window)
· legacy/indicators.py (old indicator path, unused).

## Honest limits
Simulated fills on 1m candles (touch = fill); no exchange execution; heuristic score ≠ win probability;
backtests/synthetic tests prove code behaviour, not profitability. Keep size small until real forward results exist.
