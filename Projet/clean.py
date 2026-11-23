""" in this part well clean the raw prices data and save them in just one table """
""" we'll check them for duplicates and missing values and outliers """

from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw") 
PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR /"prices.parquet"
PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]

def load_all():
    frames =[]
    for path in sorted(RAW_DIR.glob("*.csv")):
        df=pd.read_csv(path)
        df["ticker"]=path.stem
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def clean(df):
    log={"rows_read": len(df)}
    df["datetime"]=pd.to_datetime(df["datetime"])
    for column in PRICE_COLUMNS:
        df[column]=pd.to_numeric(df[column],errors="coerce")

    df=df.drop_duplicates(subset=["ticker","datetime"], keep="first")
    log["duplicates_removed"]=log["rows_read"]-len(df)

    before=len(df)
    df=df.dropna(subset=["close","volume"])
    log["missing_removed"]=before-len(df)

    before=len(df)
    valid = (df["close"] > 0) & (df["volume"] > 0) & (df["high"] >= df["low"])
    df = df[valid]
    log["invalid_removed"] = before - len(df)

    df = df.sort_values(["ticker", "datetime"]).reset_index(drop=True)
    log["rows_kept"] = len(df)
    return df, log

def check_coverage(df):
    summary=df.groupby("ticker")["datetime"].agg(["min", "max", "count"])
    print(summary.to_string())
    dates_per_ticker = df.groupby("ticker")["datetime"].apply(set)
    if len({frozenset(s) for s in dates_per_ticker})>1:
        print("WARNING: tickers do not share the same trading dates.")

def main():
    raw=load_all()
    cleaned,log=clean(raw)
    print("Cleaning log")
    for step, count in log.items():
        print(f"  {step}: {count}")
    print()
    check_coverage(cleaned)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {OUTPUT_PATH} ({len(cleaned)} rows)")


if __name__ == "__main__":
    main()