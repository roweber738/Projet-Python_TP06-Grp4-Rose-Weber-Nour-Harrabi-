import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "TTE", "NFLX", "SBUX"]
START_DATE = "2020-10-01"  # starts before 2021 so the 60-day volume baseline has history
END_DATE = "2025-12-31"
ADJUST = "all"             # split- and dividend-adjusted prices, used for returns
MAX_ROWS = 5000            # the API returns at most 5000 rows per request
PAUSE_SECONDS = 8          # free plan limit: 8 requests per minute
API_URL = "https://api.twelvedata.com/time_series"
RAW_DIR = Path("data/raw")


def download(ticker, api_key):
    """Return the daily prices of one ticker as a table."""
    params = {
        "symbol": ticker,
        "interval": "1day",
        "start_date": START_DATE,
        "end_date": END_DATE,
        "outputsize": MAX_ROWS,
        "order": "asc",
        "adjust": ADJUST,
        "apikey": api_key,
    }
    try:
        response = requests.get(API_URL, params=params, timeout=30)
    except requests.RequestException as error:
        # The error text can contain the URL, which contains the key, so print only its type.
        raise RuntimeError(f"Network error for {ticker}: {type(error).__name__}") from None

    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code} for {ticker}")

    payload = response.json()
    if payload.get("status") == "error":
        raise RuntimeError(f"API error for {ticker}: {payload.get('message')}")

    values = payload.get("values", [])
    if not values:
        raise RuntimeError(f"No data returned for {ticker}")
    if len(values) >= MAX_ROWS:
        # Reaching the limit means the data may be cut off, so stop instead of saving partial data.
        raise RuntimeError(f"{ticker} reached the {MAX_ROWS}-row limit; data may be incomplete")

    return pd.DataFrame(values)


def main():
    load_dotenv()
    api_key = os.getenv("TWELVEDATA_API_KEY")
    if not api_key:
        raise RuntimeError("TWELVEDATA_API_KEY is missing. Check your .env file.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for ticker in TICKERS:
        path = RAW_DIR / f"{ticker}.csv"
        if path.exists():
            print(f"skip  {path.name} (already downloaded)")
            continue

        df = download(ticker, api_key)
        df.to_csv(path, index=False)
        print(f"saved {path.name} ({len(df)} rows)")
        time.sleep(PAUSE_SECONDS)  # stay within the rate limit


if __name__ == "__main__":
    main()