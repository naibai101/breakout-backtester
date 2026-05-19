"""
demo.py  –  Run the BreakoutConsolidationStrategy on a handful of tickers
            and print a summary table + save individual HTML reports.
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
from backtesting import Backtest
from strategy import BreakoutConsolidationStrategy, download_data

warnings.filterwarnings("ignore")

TICKERS   = ["NVDA", "SMCI", "AXON", "CELH", "CRWD"]
START     = "2021-01-01"
END       = "2024-12-31"
CASH      = 100_000
COMM      = 0.001          # 0.1 % commission

OUTPUT_DIR = "/mnt/user-data/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

results_rows = []

print(f"\n{'='*70}")
print(f"  High-Volume Breakout + Consolidation + Weekly Bull Flag Backtester")
print(f"{'='*70}\n")

for ticker in TICKERS:
    print(f"▶  Running {ticker} …", end="  ", flush=True)
    try:
        data = download_data(ticker, start=START, end=END)
        bt   = Backtest(data, BreakoutConsolidationStrategy,
                        cash=CASH, commission=COMM, exclusive_orders=True)
        stats = bt.run()

        html_path = os.path.join(OUTPUT_DIR, f"{ticker}_backtest.html")
        bt.plot(filename=html_path, open_browser=False)

        row = {
            "Ticker"         : ticker,
            "Trades"         : int(stats["# Trades"]),
            "Win Rate %"     : round(float(stats["Win Rate [%]"]), 1),
            "Return %"       : round(float(stats["Return [%]"]), 2),
            "Max DD %"       : round(float(stats["Max. Drawdown [%]"]), 2),
            "Sharpe"         : round(float(stats["Sharpe Ratio"]), 2),
            "Avg Trade %"    : round(float(stats["Avg. Trade [%]"]), 2),
            "Best Trade %"   : round(float(stats["Best Trade [%]"]), 2),
            "Worst Trade %"  : round(float(stats["Worst Trade [%]"]), 2),
        }
        results_rows.append(row)
        print(f"✓  {row['Trades']} trades | {row['Return %']}% return | {row['Win Rate %']}% win-rate")

    except Exception as exc:
        print(f"✗  ERROR: {exc}")

if results_rows:
    df = pd.DataFrame(results_rows).set_index("Ticker")
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(df.to_string())

    csv_path = os.path.join(OUTPUT_DIR, "backtest_summary.csv")
    df.to_csv(csv_path)
    print(f"\n✓  Summary saved → {csv_path}")
    print(f"✓  Individual HTML charts saved to {OUTPUT_DIR}/")

    json_path = os.path.join(OUTPUT_DIR, "backtest_results.json")
    with open(json_path, "w") as f:
        json.dump(results_rows, f, indent=2)
    print(f"✓  JSON results  → {json_path}")

print(f"\n{'='*70}\n")
