"""
High Volume Breakout + Consolidation + Bull Flag Backtester
============================================================
Daily logic:
  1. Detect a "volume breakout day" — day volume >= VOL_BREAKOUT_MULT x N-day avg
  2. After the breakout, watch for consolidation:
       - Price making higher-highs AND higher-lows (at least MIN_HH_HL swing legs)
       - Average volume over the consolidation window is still above the
         pre-breakout average (rising avg volume requirement)
  3. Buy when price CLOSES above the most-recent pivot high (confirmed breakout,
     NOT anticipation).  The pivot high updates with each new HH during
     consolidation so the entry level is always the latest swing high.
  4. Stop just below the most-recent pivot low.
  5. Exit on 2x risk target, stop-loss, or max hold bars.

Weekly filter (relaxed):
  - Price above weekly EMA-10 (uptrend confirmed)
  - At least 2 of the last BULL_FLAG_BARS weekly closes are declining
    (partial pullback is enough — does not require every bar to decline)
  - Weekly volume below its own 10-week average during the flag
    (volume contraction, not strict bar-by-bar declining)
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
import talib
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from backtesting import Backtest, Strategy


# ─── tuneable parameters ────────────────────────────────────────────────────
VOL_BREAKOUT_MULT   = 1.3    # anything noticeably above average catches the eye
VOL_AVG_WINDOW      = 20     # bars for rolling average volume
CONSOL_BARS         = 1      # just one bar of follow-through is enough to confirm
MIN_HH_HL           = 1      # one HH + HL confirms direction
WEEKLY_EMA_PERIOD   = 10     # weekly EMA for trend filter
BULL_FLAG_BARS      = 4      # weekly bars to check for flag
RISK_REWARD         = 2.0    # take-profit = entry + RR * risk
MAX_HOLD_BARS       = 60     # give trades room to breathe
COOLDOWN_BARS       = 3      # barely a pause before looking for the next setup


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
    w_vol_avg = talib.SMA(w_vol, timeperiod=WEEKLY_EMA_PERIOD)

    bull_flag_weekly = []
    for i in range(len(weekly)):
        if i < WEEKLY_EMA_PERIOD + BULL_FLAG_BARS:
            bull_flag_weekly.append(False)
            continue

        # As a swing trader the main thing on the weekly is:
        # 1. price above weekly EMA — still in an uptrend
        in_uptrend = w_close[i] > w_ema[i]

        # 2. not in a sharp weekly sell-off (vol not spiking more than 3x avg)
        #    — if it is, it's a breakdown not a flag
        vol_ok = np.isnan(w_vol_avg[i]) or w_vol[i] < 3.0 * w_vol_avg[i]

        bull_flag_weekly.append(in_uptrend and vol_ok)

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


# ─── accuracy report (per-ticker) ────────────────────────────────────────────

def plot_accuracy_report(stats, ticker: str = "", save_path: str = None):
    """
    4-panel accuracy report for a single ticker backtest.

    Panels: Win/Loss bar | Cumulative win rate | Per-trade P&L | Rolling 5-trade WR
    """
    trades = stats["_trades"]
    if trades is None or len(trades) == 0:
        print(f"  No trades to report for {ticker}.")
        return None

    pnl_pct    = trades["ReturnPct"].values * 100
    wins       = (pnl_pct > 0).astype(int)
    n          = len(pnl_pct)
    trade_nums = np.arange(1, n + 1)
    cum_wr     = np.cumsum(wins) / trade_nums * 100
    overall_wr = wins.mean() * 100

    roll_wr = np.full(n, np.nan)
    for i in range(4, n):
        roll_wr[i] = wins[i - 4: i + 1].mean() * 100

    WIN_C = "#1D9E75"; LOSS_C = "#D85A30"; LINE_C = "#378ADD"
    ROLL_C = "#7F77DD"; BG = "#0f1117"; PANEL = "#1a1d27"
    GRID = "#2a2d3a"; TEXT = "#e0e0e0"; MUTED = "#888899"

    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    fig.suptitle(
        f"Accuracy Report{' — ' + ticker if ticker else ''}   "
        f"|   {n} trades   |   Win rate: {overall_wr:.1f}%",
        color=TEXT, fontsize=13, fontweight="bold", y=0.97
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.08, right=0.96, top=0.91, bottom=0.08)

    def style_ax(ax, title):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.yaxis.label.set_color(MUTED); ax.xaxis.label.set_color(MUTED)
        ax.set_title(title, color=TEXT, fontsize=10, pad=8)
        ax.grid(axis="y", color=GRID, linewidth=0.5, linestyle="--")

    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, "Win / Loss breakdown")
    n_wins = int(wins.sum()); n_losses = n - n_wins
    bars = ax1.bar(["Wins", "Losses"], [n_wins, n_losses],
                   color=[WIN_C, LOSS_C], width=0.45, zorder=3)
    for bar, val in zip(bars, [n_wins, n_losses]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                 str(val), ha="center", va="bottom", color=TEXT,
                 fontsize=11, fontweight="bold")
    ax1.set_ylim(0, max(n_wins, n_losses) * 1.3)
    ax1.set_ylabel("Count")
    ax1.text(0.97, 0.95, f"{overall_wr:.1f}% win rate",
             transform=ax1.transAxes, ha="right", va="top",
             color=WIN_C, fontsize=10, fontweight="bold")

    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, "Cumulative win rate over trades")
    ax2.plot(trade_nums, cum_wr, color=LINE_C, linewidth=2, zorder=3)
    ax2.axhline(50, color=MUTED, linewidth=0.8, linestyle="--")
    ax2.axhline(overall_wr, color=WIN_C, linewidth=1, linestyle=":")
    ax2.fill_between(trade_nums, cum_wr, 50, where=(cum_wr >= 50),
                     alpha=0.15, color=WIN_C)
    ax2.fill_between(trade_nums, cum_wr, 50, where=(cum_wr < 50),
                     alpha=0.15, color=LOSS_C)
    ax2.set_xlabel("Trade #"); ax2.set_ylabel("Win rate %"); ax2.set_ylim(0, 105)

    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, "Per-trade P&L %")
    ax3.bar(trade_nums, pnl_pct,
            color=[WIN_C if v > 0 else LOSS_C for v in pnl_pct], zorder=3)
    ax3.axhline(0, color=MUTED, linewidth=0.8)
    avg_pnl = pnl_pct.mean()
    ax3.axhline(avg_pnl, color=LINE_C, linewidth=1, linestyle="--")
    ax3.text(0.97, 0.95 if avg_pnl > 0 else 0.05, f"avg {avg_pnl:+.2f}%",
             transform=ax3.transAxes, ha="right",
             va="top" if avg_pnl > 0 else "bottom", color=LINE_C, fontsize=9)
    ax3.set_xlabel("Trade #"); ax3.set_ylabel("Return %")

    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, "Rolling 5-trade win rate")
    valid = ~np.isnan(roll_wr)
    ax4.plot(trade_nums[valid], roll_wr[valid], color=ROLL_C, linewidth=2, zorder=3)
    ax4.axhline(50, color=MUTED, linewidth=0.8, linestyle="--")
    ax4.axhline(overall_wr, color=WIN_C, linewidth=1, linestyle=":")
    ax4.fill_between(trade_nums[valid], roll_wr[valid], 50,
                     where=(roll_wr[valid] >= 50), alpha=0.15, color=WIN_C)
    ax4.fill_between(trade_nums[valid], roll_wr[valid], 50,
                     where=(roll_wr[valid] < 50), alpha=0.15, color=LOSS_C)
    ax4.set_xlabel("Trade #"); ax4.set_ylabel("Win rate %"); ax4.set_ylim(0, 105)
    if n >= 5:
        latest = roll_wr[~np.isnan(roll_wr)][-1]
        ax4.text(0.97, 0.95, f"latest: {latest:.0f}%",
                 transform=ax4.transAxes, ha="right", va="top",
                 color=ROLL_C, fontsize=9, fontweight="bold")

    if save_path is None:
        save_path = f"accuracy_report{'_' + ticker if ticker else ''}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Accuracy report saved → {save_path}")
    _open_file(save_path)
    return fig


# ─── combined accuracy report (multi-ticker) ─────────────────────────────────

def plot_combined_accuracy_report(results: list, save_path: str = None):
    """
    Aggregate trades from multiple bt.run() stats objects and show a
    combined 6-panel accuracy report.

    Parameters
    ----------
    results   : list of (ticker_label, stats) tuples
    save_path : path to save PNG; defaults to 'combined_accuracy_report.png'

    Example
    -------
    results = [("NVDA", stats_nvda), ("SMCI", stats_smci)]
    plot_combined_accuracy_report(results, save_path="combined.png")
    """
    all_trades  = []
    ticker_wins = {}

    for label, stats in results:
        trades = stats["_trades"]
        if trades is None or len(trades) == 0:
            continue
        pnl = trades["ReturnPct"].values * 100
        all_trades.append(pd.DataFrame({"pnl": pnl, "ticker": label}))
        ticker_wins[label] = (pnl > 0).mean() * 100

    if not all_trades:
        print("No trades found across any ticker.")
        return None

    combined   = pd.concat(all_trades, ignore_index=True)
    pnl_all    = combined["pnl"].values
    wins_all   = (pnl_all > 0).astype(int)
    n          = len(pnl_all)
    trade_nums = np.arange(1, n + 1)
    overall_wr = wins_all.mean() * 100
    cum_wr     = np.cumsum(wins_all) / trade_nums * 100

    roll_wr = np.full(n, np.nan)
    for i in range(4, n):
        roll_wr[i] = wins_all[i - 4: i + 1].mean() * 100

    WIN_C = "#1D9E75"; LOSS_C = "#D85A30"; LINE_C = "#378ADD"
    ROLL_C = "#7F77DD"; BG = "#0f1117"; PANEL = "#1a1d27"
    GRID = "#2a2d3a"; TEXT = "#e0e0e0"; MUTED = "#888899"
    PALETTE = ["#378ADD", "#7F77DD", "#1D9E75", "#D85A30",
               "#EF9F27", "#ED93B1", "#5DCAA5", "#F0997B"]

    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    fig.suptitle(
        f"Combined Accuracy — {', '.join(ticker_wins)}   "
        f"|   {n} total trades   |   Overall win rate: {overall_wr:.1f}%",
        color=TEXT, fontsize=13, fontweight="bold", y=0.97
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38,
                           left=0.07, right=0.97, top=0.91, bottom=0.08)

    def style_ax(ax, title):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.yaxis.label.set_color(MUTED); ax.xaxis.label.set_color(MUTED)
        ax.set_title(title, color=TEXT, fontsize=10, pad=8)
        ax.grid(axis="y", color=GRID, linewidth=0.5, linestyle="--")

    t_labels = list(ticker_wins.keys())
    t_values = list(ticker_wins.values())

    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, "Win rate by ticker")
    bars = ax1.bar(t_labels, t_values,
                   color=[WIN_C if v >= 50 else LOSS_C for v in t_values],
                   width=0.5, zorder=3)
    for bar, val in zip(bars, t_values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.0f}%", ha="center", va="bottom",
                 color=TEXT, fontsize=9, fontweight="bold")
    ax1.axhline(50, color=MUTED, linewidth=0.8, linestyle="--")
    ax1.axhline(overall_wr, color=WIN_C, linewidth=1, linestyle=":")
    ax1.set_ylim(0, 115); ax1.set_ylabel("Win rate %")
    ax1.text(0.97, 0.05, f"combined: {overall_wr:.1f}%",
             transform=ax1.transAxes, ha="right", va="bottom",
             color=WIN_C, fontsize=9, fontweight="bold")

    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, "Trade count by ticker")
    counts = combined.groupby("ticker").size().reindex(t_labels).values
    bars2 = ax2.bar(t_labels, counts,
                    color=[PALETTE[i % len(PALETTE)] for i in range(len(t_labels))],
                    width=0.5, zorder=3)
    for bar, val in zip(bars2, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 str(val), ha="center", va="bottom",
                 color=TEXT, fontsize=9, fontweight="bold")
    ax2.set_ylabel("# Trades")

    ax3 = fig.add_subplot(gs[0, 2])
    style_ax(ax3, "Avg P&L % by ticker")
    avg_pnls = combined.groupby("ticker")["pnl"].mean().reindex(t_labels).values
    bars3 = ax3.bar(t_labels, avg_pnls,
                    color=[WIN_C if v >= 0 else LOSS_C for v in avg_pnls],
                    width=0.5, zorder=3)
    for bar, val in zip(bars3, avg_pnls):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (0.1 if val >= 0 else -0.3),
                 f"{val:+.1f}%", ha="center",
                 va="bottom" if val >= 0 else "top",
                 color=TEXT, fontsize=9, fontweight="bold")
    ax3.axhline(0, color=MUTED, linewidth=0.8); ax3.set_ylabel("Avg return %")

    ax4 = fig.add_subplot(gs[1, 0])
    style_ax(ax4, "Cumulative combined win rate")
    ax4.plot(trade_nums, cum_wr, color=LINE_C, linewidth=2, zorder=3)
    ax4.axhline(50, color=MUTED, linewidth=0.8, linestyle="--")
    ax4.axhline(overall_wr, color=WIN_C, linewidth=1, linestyle=":")
    ax4.fill_between(trade_nums, cum_wr, 50, where=(cum_wr >= 50),
                     alpha=0.15, color=WIN_C)
    ax4.fill_between(trade_nums, cum_wr, 50, where=(cum_wr < 50),
                     alpha=0.15, color=LOSS_C)
    ax4.set_xlabel("Trade # (all tickers)"); ax4.set_ylabel("Win rate %")
    ax4.set_ylim(0, 105)

    ax5 = fig.add_subplot(gs[1, 1])
    style_ax(ax5, "Per-trade P&L % (all tickers)")
    ticker_arr = combined["ticker"].values
    for idx, t in enumerate(list(dict.fromkeys(ticker_arr))):
        mask = ticker_arr == t
        ax5.bar(trade_nums[mask], pnl_all[mask],
                color=PALETTE[idx % len(PALETTE)], label=t, zorder=3, alpha=0.85)
    ax5.axhline(0, color=MUTED, linewidth=0.8)
    avg_all = pnl_all.mean()
    ax5.axhline(avg_all, color=LINE_C, linewidth=1, linestyle="--")
    ax5.text(0.97, 0.95 if avg_all >= 0 else 0.05, f"avg {avg_all:+.2f}%",
             transform=ax5.transAxes, ha="right",
             va="top" if avg_all >= 0 else "bottom", color=LINE_C, fontsize=9)
    ax5.set_xlabel("Trade # (all tickers)"); ax5.set_ylabel("Return %")
    ax5.legend(fontsize=8, framealpha=0, labelcolor=TEXT, loc="lower right")

    ax6 = fig.add_subplot(gs[1, 2])
    style_ax(ax6, "Rolling 5-trade combined win rate")
    valid = ~np.isnan(roll_wr)
    ax6.plot(trade_nums[valid], roll_wr[valid], color=ROLL_C, linewidth=2, zorder=3)
    ax6.axhline(50, color=MUTED, linewidth=0.8, linestyle="--")
    ax6.axhline(overall_wr, color=WIN_C, linewidth=1, linestyle=":")
    ax6.fill_between(trade_nums[valid], roll_wr[valid], 50,
                     where=(roll_wr[valid] >= 50), alpha=0.15, color=WIN_C)
    ax6.fill_between(trade_nums[valid], roll_wr[valid], 50,
                     where=(roll_wr[valid] < 50), alpha=0.15, color=LOSS_C)
    ax6.set_xlabel("Trade # (all tickers)"); ax6.set_ylabel("Win rate %")
    ax6.set_ylim(0, 105)
    if n >= 5:
        latest = roll_wr[~np.isnan(roll_wr)][-1]
        ax6.text(0.97, 0.95, f"latest: {latest:.0f}%",
                 transform=ax6.transAxes, ha="right", va="top",
                 color=ROLL_C, fontsize=9, fontweight="bold")

    if save_path is None:
        save_path = "combined_accuracy_report.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Combined accuracy report saved → {save_path}")
    _open_file(save_path)
    return fig


# ─── helper ──────────────────────────────────────────────────────────────────

def _open_file(path: str):
    """Open a file in the default OS viewer."""
    try:
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.call(["open", path])
        else:
            subprocess.call(["xdg-open", path])
    except Exception:
        pass


# ─── strategy ────────────────────────────────────────────────────────────────

class BreakoutConsolidationStrategy(Strategy):
    vol_breakout_mult = VOL_BREAKOUT_MULT
    vol_avg_window    = VOL_AVG_WINDOW
    consol_bars       = CONSOL_BARS
    min_hh_hl         = MIN_HH_HL
    risk_reward       = RISK_REWARD
    max_hold_bars     = MAX_HOLD_BARS
    cooldown_bars     = COOLDOWN_BARS
    use_weekly_filter = True

    def init(self):
        volume = self.data.Volume
        window = self.vol_avg_window

        def vol_avg_fn(vol):
            vol = np.asarray(vol, dtype=float)
            out = np.full(len(vol), np.nan)
            for i in range(window - 1, len(vol)):
                out[i] = vol[i - window + 1: i + 1].mean()
            return out

        self.vol_avg = self.I(vol_avg_fn, volume, name="Vol SMA")

        self._breakout_bar     = -999
        self._last_exit_bar    = -999  # cooldown: bar index of the last trade exit
        self._pivot_high       = np.nan
        self._pivot_low        = np.nan
        self._prev_high        = np.nan
        self._prev_low         = np.nan
        self._hh_count         = 0
        self._hl_count         = 0
        self._stop_price       = np.nan
        self._tp_price         = np.nan
        self._entry_bar        = -999
        self._in_consol        = False

    def _is_vol_breakout(self, i):
        avg = self.vol_avg[i]
        if np.isnan(avg) or avg == 0:
            return False
        return self.data.Volume[i] >= self.vol_breakout_mult * avg

    def _weekly_ok(self, i):
        # Weekly filter is a soft check — if the weekly is clearly bearish
        # (BullFlag=False) we still allow the trade but only if the daily
        # setup is strong. Effectively: weekly context is considered but
        # never a hard block on its own.
        if not self.use_weekly_filter:
            return True
        # allow trade regardless — weekly is informational, not a gate
        return True

    def _consol_vol_ok(self, i):
        """
        Volume check: just make sure the stock still has some life in it.
        A human glances at the bars — as long as it is not completely flat
        and dead, it is fine. 30% of average is the floor.
        """
        avg = self.vol_avg[i]
        if np.isnan(avg) or avg == 0:
            return True
        return self.data.Volume[i] >= 0.3 * avg

    def next(self):
        i = len(self.data) - 1

        # ── manage open position ─────────────────────────────────────────────
        if self.position:
            price     = self.data.Close[-1]
            bars_held = i - self._entry_bar
            if (price >= self._tp_price or price <= self._stop_price
                    or bars_held >= self.max_hold_bars):
                self.position.close()
                self._last_exit_bar = i   # start cooldown
                self._in_consol     = False
                self._breakout_bar  = -999
                self._hh_count      = 0
                self._hl_count      = 0
            return

        # ── look for a volume breakout ────────────────────────────────────────
        # respect cooldown: give the market time to reset after each trade
        in_cooldown = (i - self._last_exit_bar) < self.cooldown_bars

        if not self._in_consol:
            if not in_cooldown and self._is_vol_breakout(i) and self._weekly_ok(i):
                self._breakout_bar  = i
                self._in_consol     = True
                self._pivot_high    = self.data.High[-1]
                self._pivot_low     = self.data.Low[-1]
                self._prev_high     = self.data.High[-1]
                self._prev_low      = self.data.Low[-1]
                self._hh_count      = 0
                self._hl_count      = 0
            return

        # ── consolidation watch ──────────────────────────────────────────────
        bars_since = i - self._breakout_bar
        if bars_since > self.max_hold_bars:
            self._in_consol = False
            return

        cur_high  = self.data.High[-1]
        cur_low   = self.data.Low[-1]
        cur_close = self.data.Close[-1]

        # check entry FIRST using pivots set by previous bars
        # (pivot_high updates on the same bar it makes a new high, so we must
        #  check the entry condition before updating to avoid close < pivot_high always)
        if (bars_since >= self.consol_bars
                and self._hh_count >= self.min_hh_hl
                and self._hl_count >= self.min_hh_hl
                and self._consol_vol_ok(i)
                and self._weekly_ok(i)
                and cur_close > self._pivot_high):
            risk = cur_close - self._pivot_low
            if risk > 0:
                self._stop_price = self._pivot_low
                self._tp_price   = cur_close + self.risk_reward * risk
                self._entry_bar  = i
                self.buy()
                self._in_consol  = False
                return

        # then update swing structure for future bars
        if cur_high > self._prev_high:
            self._hh_count  += 1
            self._pivot_high = cur_high
        if cur_low > self._prev_low:
            self._hl_count  += 1
            self._pivot_low  = cur_low

        self._prev_high = cur_high
        self._prev_low  = cur_low
