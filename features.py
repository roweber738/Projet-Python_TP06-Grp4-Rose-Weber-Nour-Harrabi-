from pathlib import Path
import numpy as np
import pandas as pd

INPUT_PATH = Path("data/processed/prices.parquet")
OUTPUT_PATH = Path("data/processed/features.parquet")
BASELINE_WINDOW = 60
LARGE_MOVE = 0.02


def add_features(df):
    df = df.sort_values(["ticker", "datetime"]).copy()
    by_ticker = df.groupby("ticker")

    df["log_return"] = by_ticker["close"].transform(lambda s: np.log(s / s.shift(1)))

    df["baseline"] = by_ticker["volume"].transform(
        lambda s: s.shift(1).rolling(BASELINE_WINDOW).median()
    )
    df["rel_volume"] = df["volume"] / df["baseline"]
    df["large_drop"] = df["log_return"] <= -LARGE_MOVE
    df["large_rise"] = df["log_return"] >= LARGE_MOVE
    return df


def check_no_look_ahead(df):
    """Check one row by hand: its baseline must be the median of the 60 days before it."""
    first_ticker = df["ticker"].iloc[0]
    sub = df[df["ticker"] == first_ticker].reset_index(drop=True)
    row = 100
    expected = sub["volume"].iloc[row - BASELINE_WINDOW : row].median()
    assert abs(sub["baseline"].iloc[row] - expected) < 1e-6, "baseline uses the wrong days"
    assert sub["baseline"].iloc[:BASELINE_WINDOW].isna().all(), "first 60 baselines should be empty"
    print("Look-ahead check passed.")


def main():
    prices = pd.read_parquet(INPUT_PATH)
    features = add_features(prices)
    check_no_look_ahead(features)

    print("\nEvents per ticker (after the first 60 days):")
    usable = features.dropna(subset=["rel_volume"])
    counts = usable.groupby("ticker")[["large_drop", "large_rise"]].sum()
    print(counts.to_string())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {OUTPUT_PATH} ({len(features)} rows)")


if __name__ == "__main__":
    main()