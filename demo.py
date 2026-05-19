"""
demo.py  -  Run the BreakoutConsolidationStrategy on a list of tickers
            using real price data from yfinance.

Outputs per ticker:
  - {TICKER}_backtest.html         interactive equity curve
  - {TICKER}_accuracy_report.png   per-ticker accuracy panels

Outputs overall:
  - combined_accuracy_report.png   combined win rate across all tickers
  - backtest_summary.csv
"""

import os
import json
import warnings
import yfinance as yf
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
accuracy_input = []

print(f"\n{'='*70}")
print(f"  High-Volume Breakout + Consolidation + Weekly Bull Flag Backtester")
print(f"{'='*70}\n")

for ticker in TICKERS:
    print(f"Running {ticker} ...", end="  ", flush=True)
    try:
        data = download_data(ticker, start=START, end=END)

        if len(data) < 100:
            print(f"SKIPPED — not enough data returned ({len(data)} bars)")
            continue

        bt    = Backtest(data, BreakoutConsolidationStrategy,
                         cash=CASH, commission=COMM, exclusive_orders=True)
        stats = bt.run()

        n_trades = int(stats["# Trades"])
        if n_trades == 0:
            print("SKIPPED — no trades found for this ticker in the date range")
            continue

        # interactive equity-curve chart
        html_path = os.path.join(OUTPUT_DIR, f"{ticker}_backtest.html")
        bt.plot(filename=html_path, open_browser=False)

        # per-ticker accuracy PNG (auto-opens)
        acc_path = os.path.join(OUTPUT_DIR, f"{ticker}_accuracy_report.png")
        plot_accuracy_report(stats, ticker=ticker, save_path=acc_path)

        row = {
            "Ticker"        : ticker,
            "Trades"        : n_trades,
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
        print(f"{n_trades} trades | {row['Return %']}% return | {row['Win Rate %']}% win-rate")

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

    if len(accuracy_input) > 1:
        combined_path = os.path.join(OUTPUT_DIR, "combined_accuracy_report.png")
        plot_combined_accuracy_report(accuracy_input, save_path=combined_path)
else:
    print("\n  No results — check your tickers and internet connection.")

print(f"\n{'='*70}\n")
