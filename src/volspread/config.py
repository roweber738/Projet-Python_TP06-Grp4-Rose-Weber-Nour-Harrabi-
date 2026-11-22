"""Central configuration for the volume-asymmetry study.

Every parameter that affects the results lives here. Change a value in this
file and the rest of the project follows; do not hard-code numbers elsewhere.
"""

# Stocks studied (all US-listed). TTE is a Paris company listed as an ADR on
# the NYSE, so its volume is only part of total trading; this caveat goes in
# the report.
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "TTE", "NFLX", "SBUX"]

# Data are downloaded from WARMUP_START because the volume baseline needs about
# 60 trading days of history before the analysis period begins.
WARMUP_START = "2020-10-01"
ANALYSIS_START = "2021-01-01"
ANALYSIS_END = "2025-12-31"

# Relative volume: today's volume divided by the median of the previous
# BASELINE_WINDOW trading days. Today is excluded to avoid look-ahead bias.
BASELINE_WINDOW = 60

# Large-move thresholds, applied to the absolute log return.
LARGE_MOVE_THRESHOLD = 0.02       # 2%: primary threshold
VERY_LARGE_MOVE_THRESHOLD = 0.03  # 3%
NORMAL_DAY_THRESHOLD = 0.01       # |r| below 1% counts as a normal day

# Statistics
BOOTSTRAP_RESAMPLES = 10_000
RANDOM_SEED = 42

# Folders, relative to the project root (run all commands from the root)
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
FIGURES_DIR = "reports/figures"