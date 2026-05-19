"""
High Volume Breakout + Consolidation + Bull Flag Backtester
============================================================
Daily logic:
  1. Detect a "volume breakout day" — day volume >= VOL_BREAKOUT_MULT × N-day avg
  2. After the breakout, watch for consolidation:
       - Price making higher-highs AND higher-lows (at least MIN_HH_HL swing legs)
       - Rolling average volume still rising (slope > 0)
  3. Buy when price CLOSES above the breakout-day pivot high (confirmed breakout,
     NOT anticipation — the frozen high from the breakout bar is the trigger)
  4. Stop just below the breakout-day pivot low.
  5. Exit on 2x risk target, stop-loss, or max hold bars.

Weekly filter:
  - Only take trades where the weekly chart shows a bull flag:
       * Price above weekly EMA-10 (uptrend)
       * Last BULL_FLAG_BARS weekly closes declining (flag / pullback)
       * Volume contracting during the flag
"""

import numpy as np
import pandas as pd
import talib
import yfinance as yf
from backtesting import Backtest, Strategy


# ─── tuneable parameters ────────────────────────────────────────────────────
VOL_BREAKOUT_MULT   = 2.0
VOL_AVG_WINDOW      = 20
CONSOL_BARS         = 5
MIN_HH_HL           = 2
WEEKLY_EMA_PERIOD   = 10
BULL_FLAG_BARS      = 4
RISK_REWARD         = 2.0
MAX_HOLD_BARS       = 40


def download_data(ticker: str, start: str = "2020-01-01", end: str = "2024-12-31"):
    """Download daily OHLCV and add a weekly bull-flag boolean column."""
    daily = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = daily.columns.get_level_values(0)
    daily = daily[["Open", "High", "Low", "Close", "Volume"]].dropna()

    weekly = yf.download(ticker, start=start, end=end, interval="1wk",
                         auto_adjust=True, progress=False)
    if isinstance(weekly.columns, pd.MultiIndex):
        weekly.columns = weekly.columns.get_level_values(0)
    weekly = weekly[["Open", "High", "Low", "Close", "Volume"]].dropna()

    w_close = weekly["Close"].values.astype(float)
    w_vol   = weekly["Volume"].values.astype(float)
    w_ema   = talib.EMA(w_close, timeperiod=WEEKLY_EMA_PERIOD)

    bull_flag_weekly = []
    for i in range(len(weekly)):
        if i < WEEKLY_EMA_PERIOD + BULL_FLAG_BARS:
            bull_flag_weekly.append(False)
            continue
        in_uptrend   = w_close[i] > w_ema[i]
        recent_c     = w_close[i - BULL_FLAG_BARS: i]
        declining    = all(recent_c[j] >= recent_c[j + 1] for j in range(len(recent_c) - 1))
        recent_v     = w_vol[i - BULL_FLAG_BARS: i]
        vol_contract = all(recent_v[j] >= recent_v[j + 1] for j in range(len(recent_v) - 1))
        bull_flag_weekly.append(in_uptrend and declining and vol_contract)

    weekly["BullFlag"] = bull_flag_weekly
    daily["BullFlag"]  = False

    weekly.index = pd.to_datetime(weekly.index).tz_localize(None)
    daily.index  = pd.to_datetime(daily.index).tz_localize(None)

    for i, wdate in enumerate(weekly.index):
        next_wdate = weekly.index[i + 1] if i + 1 < len(weekly) \
                     else daily.index[-1] + pd.Timedelta(days=7)
        mask = (daily.index >= wdate) & (daily.index < next_wdate)
        daily.loc[mask, "BullFlag"] = weekly.iloc[i]["BullFlag"]

    return daily


class BreakoutConsolidationStrategy(Strategy):
    vol_breakout_mult = VOL_BREAKOUT_MULT
    vol_avg_window    = VOL_AVG_WINDOW
    consol_bars       = CONSOL_BARS
    min_hh_hl         = MIN_HH_HL
    risk_reward       = RISK_REWARD
    max_hold_bars     = MAX_HOLD_BARS
    use_weekly_filter = True

    def init(self):
        volume = self.data.Volume

        self.vol_avg = self.I(talib.SMA, volume, self.vol_avg_window,
                              name="Vol SMA")

        def vol_slope(vol):
            sma   = talib.SMA(vol, self.vol_avg_window)
            slope = np.full_like(sma, np.nan)
            for i in range(5, len(sma)):
                if not (np.isnan(sma[i]) or np.isnan(sma[i - 5])):
                    slope[i] = sma[i] - sma[i - 5]
            return slope

        self.vol_slope = self.I(vol_slope, volume, name="Vol Slope")

        self._breakout_bar       = -999
        self._entry_pivot_high   = np.nan
        self._entry_pivot_low    = np.nan
        self._prev_high          = np.nan
        self._prev_low           = np.nan
        self._hh_count           = 0
        self._hl_count           = 0
        self._stop_price         = np.nan
        self._tp_price           = np.nan
        self._entry_bar          = -999
        self._in_consol          = False

    def _is_vol_breakout(self, i):
        avg = self.vol_avg[i]
        if np.isnan(avg) or avg == 0:
            return False
        return self.data.Volume[i] >= self.vol_breakout_mult * avg

    def _weekly_ok(self, i):
        return (not self.use_weekly_filter) or bool(self.data.BullFlag[i])

    def next(self):
        i = len(self.data) - 1

        if self.position:
            price     = self.data.Close[-1]
            bars_held = i - self._entry_bar
            if price >= self._tp_price or price <= self._stop_price \
                    or bars_held >= self.max_hold_bars:
                self.position.close()
                self._in_consol    = False
                self._breakout_bar = -999
                self._hh_count     = 0
                self._hl_count     = 0
            return

        if not self._in_consol:
            if self._is_vol_breakout(i) and self._weekly_ok(i):
                self._breakout_bar     = i
                self._in_consol        = True
                self._entry_pivot_high = self.data.High[-1]
                self._entry_pivot_low  = self.data.Low[-1]
                self._prev_high        = self.data.High[-1]
                self._prev_low         = self.data.Low[-1]
                self._hh_count         = 0
                self._hl_count         = 0
            return

        bars_since = i - self._breakout_bar
        if bars_since > self.max_hold_bars:
            self._in_consol = False
            return

        cur_high  = self.data.High[-1]
        cur_low   = self.data.Low[-1]
        cur_close = self.data.Close[-1]

        if cur_high > self._prev_high:
            self._hh_count += 1
        if cur_low > self._prev_low:
            self._hl_count += 1
        self._prev_high = cur_high
        self._prev_low  = cur_low

        if bars_since < self.consol_bars:
            return
        if self._hh_count < self.min_hh_hl or self._hl_count < self.min_hh_hl:
            return

        vs = self.vol_slope[i]
        if np.isnan(vs) or vs <= 0:
            return

        if not self._weekly_ok(i):
            return

        if cur_close > self._entry_pivot_high:
            risk = cur_close - self._entry_pivot_low
            if risk <= 0:
                return
            self._stop_price = self._entry_pivot_low
            self._tp_price   = cur_close + self.risk_reward * risk
            self._entry_bar  = i
            self.buy()
            self._in_consol = False
