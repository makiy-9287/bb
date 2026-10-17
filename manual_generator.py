"""
Weekend Manual Data Generator with Auto-Upload (Multi-Service Fallback)
Fetches raw data for Top N coins, generates JSON, and uploads to a cloud host.
Returns a direct download link in the terminal.
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
TOP_COINS_LIMIT = 60   # ඔයාට ඕන ගාන මෙතන දාන්න

TF_LIMITS = {
    "4h": 300,
    "1h": 300,
    "15m": 100,
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
    except Exception:
        return []

def upload_file(file_path: str) -> str:
    """
    Tries multiple free file hosts. Returns the first successful URL.
    """
    # Service 1: Catbox.moe (Permanent)
    logger.info("Trying upload to catbox.moe...")
    try:
        res = subprocess.run(
            ["curl", "-s", "-X", "POST", "https://catbox.moe/user/api.php",
             "-F", "reqtype=fileupload",
             "-F", f"fileToUpload=@{file_path}"],
            capture_output=True, text=True, timeout=120
        )
        url = res.stdout.strip()
        if res.returncode == 0 and url.startswith("https://"):
            return url
        logger.warning(f"Catbox failed: {url[:100]}")
    except Exception as e:
        logger.warning(f"Catbox error: {e}")

    # Service 2: Litterbox (Temporary, up to 72h)
    logger.info("Trying upload to litterbox...")
    try:
        res = subprocess.run(
            ["curl", "-s", "-X", "POST", "https://litterbox.catbox.moe/resources/internals/api.php",
             "-F", "reqtype=fileupload",
             "-F", "time=72h",
             "-F", f"fileToUpload=@{file_path}"],
            capture_output=True, text=True, timeout=120
        )
        url = res.stdout.strip()
        if res.returncode == 0 and url.startswith("https://"):
            return url
        logger.warning(f"Litterbox failed: {url[:100]}")
    except Exception as e:
        logger.warning(f"Litterbox error: {e}")

    # Service 3: EasySend (No auth, JSON response)
    logger.info("Trying upload to easysend.co...")
    try:
        res = subprocess.run(
            ["curl", "-s", "-X", "POST", "https://easysend.co/api/v1/upload",
             "-F", f"files[]=@{file_path}"],
            capture_output=True, text=True, timeout=120
        )
        if res.returncode == 0:
            data = json.loads(res.stdout)
            if data.get("success") and data.get("share_url"):
                return f"https://easysend.co{data['share_url']}"
        logger.warning(f"EasySend failed: {res.stdout[:100]}")
    except Exception as e:
        logger.warning(f"EasySend error: {e}")

    return None

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

    logger.info("Fetching OHLCV data (this will take a few minutes for 100 coins)...")
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
    
    # ── UPLOAD ─────────────────────────────────────────────────────
    logger.info("Uploading to cloud... (please wait)")
    link = upload_file(OUTPUT_FILE)
    
    if link:
        logger.info("\n" + "="*70)
        logger.info(f"🎉 DOWNLOAD LINK: {link}")
        logger.info("="*70 + "\n")
        logger.info("📌 Copy the link above and download the file on your phone/PC.")
    else:
        logger.error("❌ All upload services failed. Your VPS IP might be blocked.")
        logger.info("💡 Alternative: You can download the file via SCP command:")
        logger.info(f"   scp root@{os.popen('hostname -I').read().split()[0]}:~/bb/{OUTPUT_FILE} .")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
