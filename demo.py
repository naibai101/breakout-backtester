"""
demo.py  -  Run the BreakoutConsolidationStrategy on a list of tickers,
            print a summary table, save individual HTML charts, and
            generate both per-ticker and combined accuracy report PNGs.

Usage
-----
1. Add your tickers to the TICKERS list below.
2. Run:  python demo.py
3. Per-ticker accuracy PNGs and a combined report will open automatically.

Note: yfinance requires internet access. If running offline, replace
      download_data() calls with your own OHLCV DataFrames that include
      a boolean 'BullFlag' column.
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
from backtesting import Backtest
from strategy import (BreakoutConsolidationStrategy, download_data,
                      plot_accuracy_report, plot_combined_accuracy_report)

warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────────────────────
TICKERS    = ["NVDA", "SMCI", "AXON", "CELH", "CRWD"]
START      = "2021-01-01"
END        = "2024-12-31"
CASH       = 100_000
COMM       = 0.001        # 0.1% commission per trade
OUTPUT_DIR = "."          # all output files saved to current folder

os.makedirs(OUTPUT_DIR, exist_ok=True)

results_rows   = []
accuracy_input = []   # list of (ticker, stats) for the combined report

print(f"\n{'='*70}")
print(f"  High-Volume Breakout + Consolidation + Weekly Bull Flag Backtester")
print(f"{'='*70}\n")

for ticker in TICKERS:
    print(f"Running {ticker} ...", end="  ", flush=True)
    try:
        data  = download_data(ticker, start=START, end=END)
        bt    = Backtest(data, BreakoutConsolidationStrategy,
                         cash=CASH, commission=COMM, exclusive_orders=True)
        stats = bt.run()

        # interactive equity-curve chart (open in browser manually)
        html_path = os.path.join(OUTPUT_DIR, f"{ticker}_backtest.html")
        bt.plot(filename=html_path, open_browser=False)

        # per-ticker accuracy PNG (auto-opens in default image viewer)
        acc_path = os.path.join(OUTPUT_DIR, f"{ticker}_accuracy_report.png")
        plot_accuracy_report(stats, ticker=ticker, save_path=acc_path)

        row = {
            "Ticker"        : ticker,
            "Trades"        : int(stats["# Trades"]),
            "Win Rate %"    : round(float(stats["Win Rate [%]"]), 1),
            "Return %"      : round(float(stats["Return [%]"]), 2),
            "Max DD %"      : round(float(stats["Max. Drawdown [%]"]), 2),
            "Sharpe"        : round(float(stats["Sharpe Ratio"]), 2),
            "Avg Trade %"   : round(float(stats["Avg. Trade [%]"]), 2),
            "Best Trade %"  : round(float(stats["Best Trade [%]"]), 2),
            "Worst Trade %" : round(float(stats["Worst Trade [%]"]), 2),
        }
        results_rows.append(row)
        accuracy_input.append((ticker, stats))
        print(f"{row['Trades']} trades | {row['Return %']}% return | {row['Win Rate %']}% win-rate")

    except Exception as exc:
        print(f"ERROR: {exc}")

# ── summary ───────────────────────────────────────────────────────────────────
if results_rows:
    df = pd.DataFrame(results_rows).set_index("Ticker")
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(df.to_string())

    df.to_csv(os.path.join(OUTPUT_DIR, "backtest_summary.csv"))

    with open(os.path.join(OUTPUT_DIR, "backtest_results.json"), "w") as f:
        json.dump(results_rows, f, indent=2)

    print(f"\n  Files saved to: {os.path.abspath(OUTPUT_DIR)}/")

    # combined accuracy report across all tickers (auto-opens)
    if len(accuracy_input) > 1:
        combined_path = os.path.join(OUTPUT_DIR, "combined_accuracy_report.png")
        plot_combined_accuracy_report(accuracy_input, save_path=combined_path)
else:
    print("\n  No results — check your tickers and internet connection.")

print(f"\n{'='*70}\n")
