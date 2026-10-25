Commands run in the build sandbox (Python 3, pytest 9.1.1):
  python -m pytest tests -q          → 68 passed
  python -m pyflakes *.py            → clean except 1 unused local (ict_engine.py seg_cl, harmless)
  python backtest.py --synthetic     → harness runs; 0 trades on random synthetic data (engine is strict)
NOT run: mypy/ruff, any live Binance/OpenRouter/Telegram call (sandbox has no access), real-data backtest.
