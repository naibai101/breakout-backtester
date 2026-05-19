# Breakout Consolidation Backtester

A Python backtester for a momentum strategy that combines a **daily high-volume breakout + consolidation** setup with a **weekly bull flag filter**, built on [backtesting.py](https://kernc.github.io/backtesting.py/) and [TA-Lib](https://ta-lib.org/).

\---

## Strategy Logic

### Daily timeframe

1. **Volume breakout** — flags a bar when volume is ≥ `VOL\_BREAKOUT\_MULT` × 20-day average volume. The high and low of that bar are frozen as the pivot levels.
2. **Consolidation** — after the breakout bar, waits for:

   * At least `MIN\_HH\_HL` higher-highs **and** higher-lows (structural confirmation)
   * 20-day average volume still rising (positive 5-bar slope)
   * Minimum `CONSOL\_BARS` bars elapsed
3. **Confirmed entry** — buys on close above the frozen breakout-day pivot high. No anticipation — the level must be cleared on close.
4. **Stop \& target** — stop placed at the breakout-day pivot low; take-profit at `RISK\_REWARD × risk` above entry.

### Weekly timeframe (filter)

Only takes a trade when the weekly chart shows a bull flag:

* Price above the weekly EMA-10 (uptrend)
* Last 4 weekly closes declining (flag / pullback)
* Weekly volume contracting during the flag

\---

## Parameters

|Parameter|Default|Description|
|-|-|-|
|`VOL\_BREAKOUT\_MULT`|`2.0`|Volume multiple vs 20-day avg to qualify as a breakout|
|`VOL\_AVG\_WINDOW`|`20`|Rolling window for average volume|
|`CONSOL\_BARS`|`5`|Minimum bars to wait before entry|
|`MIN\_HH\_HL`|`2`|Minimum higher-high / higher-low legs required|
|`WEEKLY\_EMA\_PERIOD`|`10`|Weekly EMA period for trend filter|
|`BULL\_FLAG\_BARS`|`4`|Weekly bars to check for declining closes/volume|
|`RISK\_REWARD`|`2.0`|Take-profit as a multiple of risk|
|`MAX\_HOLD\_BARS`|`40`|Force-exit after this many bars|
|`use\_weekly\_filter`|`True`|Toggle weekly bull-flag filter on/off|

\---

## Installation

```bash
pip install backtesting ta-lib yfinance pandas numpy
```

> TA-Lib requires the underlying C library. See the \[TA-Lib installation guide](https://github.com/TA-Lib/ta-lib-python#installation) for your platform.

\---

## Usage

### Run the demo

```bash
python demo.py
```

The demo runs against 5 synthetic momentum-style tickers and saves per-ticker interactive HTML charts plus a `backtest\_summary.csv` to the working directory.

### Use with real tickers

In `demo.py`, replace the `configs` list with your own tickers using `download\_data()`:

```python
from strategy import BreakoutConsolidationStrategy, download\_data
from backtesting import Backtest

data = download\_data("NVDA", start="2021-01-01", end="2024-12-31")

bt = Backtest(data, BreakoutConsolidationStrategy,
              cash=100\_000, commission=0.001, exclusive\_orders=True)
stats = bt.run()
print(stats)
bt.plot()
```

### Parameter optimisation

backtesting.py's built-in grid search:

```python
stats, heatmap = bt.optimize(
    vol\_breakout\_mult=\[1.5, 2.0, 2.5, 3.0],
    consol\_bars=\[3, 5, 8],
    min\_hh\_hl=\[1, 2, 3],
    risk\_reward=\[1.5, 2.0, 2.5],
    maximize="Sharpe Ratio",
    return\_heatmap=True
)
```

\---

## Output

Each run produces:

* `{TICKER}\_backtest.html` — interactive equity curve, trade markers, and volume indicators (powered by Bokeh)
* `backtest\_summary.csv` — aggregated stats across all tickers

\---

## File Structure

```
breakout-backtester/
├── strategy.py     # Strategy class + download\_data() helper
├── demo.py         # Demo runner (synthetic data by default)
├── requirements.txt
└── README.md
```



