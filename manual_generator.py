"""
Weekend Manual Data Generator with Auto-Upload
Fetches raw data for Top 100 coins, generates JSON, and uploads it.
Returns a direct download link.
"""
import asyncio
import json
import logging
import os
import subprocess
from datetime import datetime
from typing import List

import aiohttp
import ccxt.async_support as ccxt

# ── CONFIGURATION ────────────────────────────────────────────────
TOP_COINS_LIMIT = 100   # ඔයාට ඕන ගාන මෙතන දාන්න

TF_LIMITS = {
    "4h": 400,
    "1h": 300,
    "15m": 200,
    "5m": 50
}
OUTPUT_FILE = "manual_data_100.json"
BINANCE_FUTURES_REST = "https://fapi.binance.com"

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-7s │ %(message)s")
logger = logging.getLogger("manual_gen")

async def fetch_top_symbols(limit: int) -> List[str]:
    url = f"{BINANCE_FUTURES_REST}/fapi/v1/ticker/24hr"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=15) as resp:
            data = await resp.json()
    
    usdt = [t for t in data if t["symbol"].endswith("USDT") and t["symbol"] != "USDCUSDT"]
    usdt.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    return [t["symbol"] for t in usdt[:limit]]

async def fetch_ohlcv(exchange, symbol: str, timeframe: str, limit: int):
    try:
        raw = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        # HLCV Format: [high, low, close, volume]
        return [
            [round(c[2], 6), round(c[3], 6), round(c[4], 6), round(c[5], 2)]
            for c in raw
        ]
    except Exception as e:
        return []

async def main():
    logger.info(f"Fetching Top {TOP_COINS_LIMIT} symbols...")
    symbols = await fetch_top_symbols(TOP_COINS_LIMIT)
    
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    
    sem = asyncio.Semaphore(10) 
    
    async def process_coin(sym):
        async with sem:
            coin_data = {}
            for tf, limit in TF_LIMITS.items():
                candles = await fetch_ohlcv(exchange, sym, tf, limit)
                if candles:
                    coin_data[tf] = candles
            if coin_data:
                logger.info(f"Downloaded: {sym}")
                return sym, coin_data
            return sym, None

    logger.info("Fetching OHLCV data (this will take a few minutes)...")
    tasks = [process_coin(s) for s in symbols]
    results = await asyncio.gather(*tasks)
    
    await exchange.close()

    final_data = {
        "meta": {
            "timestamp": datetime.utcnow().isoformat(),
            "format": "[high, low, close, volume] - oldest first",
            "timeframes": TF_LIMITS
        },
        "coins": {}
    }
    
    for sym, data in results:
        if data:
            final_data["coins"][sym] = data

    with open(OUTPUT_FILE, "w") as f:
        json.dump(final_data, f, separators=(",", ":"))

    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    logger.info(f"✅ Generated {OUTPUT_FILE} ({size_mb:.1f} MB) for {len(final_data['coins'])} coins.")
    
    # ── AUTO UPLOAD TO CLOUD ─────────────────────────────────────
    logger.info("Uploading file to cloud to get a direct link...")
    try:
        # bashupload.com is a free, reliable file sharing service
        result = subprocess.run(
            ["curl", "bashupload.com", "-T", OUTPUT_FILE],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0 and result.stdout:
            link = result.stdout.strip()
            logger.info("\n" + "="*60)
            logger.info(f"🎉 DOWNLOAD LINK: {link}")
            logger.info("="*60 + "\n")
            logger.info("Copy this link and download the file. (Link expires in 3 days)")
        else:
            logger.error(f"Upload failed. Please try again. Error: {result.stderr}")
    except Exception as e:
        logger.error(f"Upload error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
