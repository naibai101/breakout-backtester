"""
demo.py  -  Run the BreakoutConsolidationStrategy on a list of tickers
            using real price data from yfinance.

Outputs per ticker:
  - {TICKER}_backtest.html         interactive equity curve (open in browser)
  - {TICKER}_accuracy_report.png   per-ticker accuracy panels
  - {TICKER}_trades.csv            every trade with entry, exit, return, duration
                                   and which exit condition fired

Outputs overall:
  - combined_accuracy_report.png   combined win rate across all tickers
  - backtest_summary.csv           high-level stats per ticker
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
TICKERS    = ["NVDA", "AMZN", "TSLA", "AAPL", "CRWD", "AMD", "TPL", "NFLX"]
START      = "2000-01-01"
END        = "2024-12-31"
CASH       = 100_000
COMM       = 0.001
OUTPUT_DIR = "."

os.makedirs(OUTPUT_DIR, exist_ok=True)

results_rows   = []
accuracy_input = []

print(f"\n{'='*70}")
print(f"  High-Volume Breakout + Consolidation + Weekly Bull Flag Backtester")
print(f"{'='*70}\n")

for ticker in TICKERS:
    print(f"\nRunning {ticker} ...", flush=True)
    try:
        data = download_data(ticker, start=START, end=END)

        if len(data) < 100:
            print(f"  SKIPPED — not enough data ({len(data)} bars)")
            continue

        bt    = Backtest(data, BreakoutConsolidationStrategy,
                         cash=CASH, commission=COMM, exclusive_orders=True)
        stats = bt.run()

        n_trades = int(stats["# Trades"])
        if n_trades == 0:
            print("  SKIPPED — no trades found in this date range")
            continue

        # ── interactive chart (open in browser to see every trade on the chart)
        html_path = os.path.join(OUTPUT_DIR, f"{ticker}_backtest.html")
        bt.plot(filename=html_path, open_browser=False)
        print(f"  Chart saved → {ticker}_backtest.html  (open this in your browser)")

        # ── accuracy PNG
        acc_path = os.path.join(OUTPUT_DIR, f"{ticker}_accuracy_report.png")
        plot_accuracy_report(stats, ticker=ticker, save_path=acc_path)

        # ── trade-by-trade breakdown ──────────────────────────────────────────
        trades = stats["_trades"].copy()

        # label which exit condition fired for each trade
        def label_exit(row):
            ret = row["ReturnPct"] * 100
            dur = int(row["Duration"].days) if hasattr(row["Duration"], "days") else int(row["Duration"])

            if dur >= 5:
                return "⏰ time exit (5d)"
            if dur == 1 and ret > 0:
                return "✅ day-1 green exit"
            if ret <= -3.0:
                return "🛑 hard stop / trail"
            if ret < 0 and dur <= 3:
                return "🚪 no follow-through (3d)"
            if ret > 0:
                return "📈 trail / late exit"
            return "🚪 no follow-through (3d)"

        trades["Exit Reason"] = trades.apply(label_exit, axis=1)
        trades["Return %"]    = (trades["ReturnPct"] * 100).round(2)
        trades["Days Held"]   = trades["Duration"].apply(
            lambda d: d.days if hasattr(d, "days") else int(d)
        )

        display_cols = ["EntryTime", "ExitTime", "Days Held",
                        "Return %", "Exit Reason"]
        display_cols = [c for c in display_cols if c in trades.columns]

        print(f"\n  {'─'*60}")
        print(f"  {ticker} — {n_trades} trades")
        print(f"  {'─'*60}")
        print(trades[display_cols].to_string(index=False))

        # exit reason summary
        reason_counts = trades["Exit Reason"].value_counts()
        print(f"\n  Exit breakdown:")
        for reason, count in reason_counts.items():
            pct = count / n_trades * 100
            print(f"    {reason:<30} {count:>3} trades  ({pct:.0f}%)")

        # win/loss by exit type
        print(f"\n  Avg return by exit type:")
        for reason in reason_counts.index:
            avg = trades.loc[trades["Exit Reason"] == reason, "Return %"].mean()
            print(f"    {reason:<30} avg {avg:+.2f}%")

        print(f"  {'─'*60}")

        # save trades to CSV
        csv_path = os.path.join(OUTPUT_DIR, f"{ticker}_trades.csv")
        trades[display_cols + ["Return %", "Exit Reason"]].to_csv(csv_path, index=False)
        print(f"  Trades saved → {ticker}_trades.csv")

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

    except Exception as exc:
        print(f"  ERROR: {exc}")

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
