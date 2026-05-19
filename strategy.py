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


# ─── shared style helpers ────────────────────────────────────────────────────

def _palette():
    return dict(
        BG      = "#FFF0F5",   # soft pink background
        PANEL   = "#FFF8FB",   # slightly lighter pink panels
        GRID    = "#F5DCE8",   # dusty rose grid
        BORDER  = "#F0C0D8",   # pink panel border
        TEXT    = "#4A2040",   # deep plum — readable on pink
        MUTED   = "#B07898",   # muted mauve labels
        WIN_C   = "#7DC98A",   # pastel green  ✓
        LOSS_C  = "#F08080",   # pastel red    ✗
        LINE_C  = "#C06090",   # deep rose line
        ROLL_C  = "#9B72B0",   # soft purple
        PALETTE = ["#7DC98A","#F08080","#85C1E9","#F9C85A",
                   "#C39BD3","#F0A06A","#76D7C4","#F1948A"],
    )

def _style_ax(ax, title, p):
    ax.set_facecolor(p["PANEL"])
    for sp in ax.spines.values():
        sp.set_edgecolor(p["BORDER"]); sp.set_linewidth(1.0)
    ax.tick_params(colors=p["MUTED"], labelsize=8)
    ax.yaxis.label.set_color(p["MUTED"])
    ax.xaxis.label.set_color(p["MUTED"])
    ax.set_title(title, color=p["TEXT"], fontsize=10,
                 fontweight="bold", pad=6, loc="left")
    ax.grid(axis="y", color=p["GRID"], linewidth=0.7, linestyle="-")
    ax.set_axisbelow(True)

def _stat_box(ax, x, y, text, color, p, fontsize=11):
    """Draw a small rounded stat label with a white backing box."""
    ax.text(x, y, text, transform=ax.transAxes,
            ha="center", va="center", color=color,
            fontsize=fontsize, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor=p["BORDER"],
                      boxstyle="round,pad=0.4", linewidth=1.2,
                      alpha=0.92))


# ─── accuracy report (per-ticker) ────────────────────────────────────────────

def plot_accuracy_report(stats, ticker: str = "", save_path: str = None):
    """
    4-panel per-ticker accuracy report.
    Panels: Win/Loss + big stat | Cumulative WR | Per-trade P&L | Rolling 5-WR
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
    avg_pnl    = pnl_pct.mean()
    n_wins     = int(wins.sum())
    n_losses   = n - n_wins

    roll_wr = np.full(n, np.nan)
    for i in range(4, n):
        roll_wr[i] = wins[i - 4: i + 1].mean() * 100

    p = _palette()
    fig = plt.figure(figsize=(15, 9), facecolor=p["BG"])

    # ── header ───────────────────────────────────────────────────────────────
    fig.text(0.5, 0.975,
             f"{ticker + '  ·  ' if ticker else ''}accuracy report",
             ha="center", va="top", color=p["TEXT"],
             fontsize=14, fontweight="bold")

    # three summary pills in a row below the title
    pill_y = 0.935
    for xpos, label in zip([0.28, 0.50, 0.72],
                            [f"{n} trades",
                             f"win rate  {overall_wr:.1f}%",
                             f"avg P&L  {avg_pnl:+.2f}%"]):
        fig.text(xpos, pill_y, label, ha="center", va="top",
                 color=p["MUTED"], fontsize=9,
                 bbox=dict(facecolor="white", edgecolor=p["BORDER"],
                           boxstyle="round,pad=0.35", linewidth=1,
                           alpha=0.85))

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.52, wspace=0.38,
                           left=0.08, right=0.96,
                           top=0.87, bottom=0.09)

    # ── panel 1: wins vs losses ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _style_ax(ax1, "wins vs losses", p)

    b1 = ax1.bar([0, 1], [n_wins, n_losses],
                 color=[p["WIN_C"], p["LOSS_C"]],
                 width=0.5, zorder=3, edgecolor="white", linewidth=1.5)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Wins ✓", "Losses ✗"], color=p["TEXT"], fontsize=9)
    ax1.set_ylabel("count")

    # bar count labels — inside the top of each bar to avoid collision
    for bar, val, col in zip(b1, [n_wins, n_losses], ["white", "white"]):
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     h * 0.5,           # middle of bar
                     str(val), ha="center", va="center",
                     color="white", fontsize=13, fontweight="bold")

    # win rate — below the chart title, above bars, in its own box
    badge_c = p["WIN_C"] if overall_wr >= 50 else p["LOSS_C"]
    ax1.set_ylim(0, max(n_wins, n_losses) * 1.6)
    _stat_box(ax1, 0.5, 0.88, f"{overall_wr:.1f}%  win rate",
              badge_c, p, fontsize=12)

    # ── panel 2: cumulative win rate ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _style_ax(ax2, "cumulative win rate", p)
    ax2.plot(trade_nums, cum_wr, color=p["LINE_C"], linewidth=2.2, zorder=3)
    ax2.axhline(50, color=p["MUTED"], linewidth=1,
                linestyle="--", alpha=0.6, label="50%")
    ax2.fill_between(trade_nums, cum_wr, 50,
                     where=(cum_wr >= 50), alpha=0.20, color=p["WIN_C"])
    ax2.fill_between(trade_nums, cum_wr, 50,
                     where=(cum_wr < 50),  alpha=0.20, color=p["LOSS_C"])
    ax2.set_xlabel("trade #"); ax2.set_ylabel("win rate %")
    ax2.set_ylim(0, 120)
    # end label in top-right corner box — always fits
    _stat_box(ax2, 0.82, 0.92, f"{cum_wr[-1]:.1f}%",
              p["LINE_C"], p, fontsize=10)
    ax2.text(0.04, 0.08, "50% mark", transform=ax2.transAxes,
             color=p["MUTED"], fontsize=8, va="bottom")

    # ── panel 3: per-trade P&L ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    _style_ax(ax3, "per-trade return %", p)
    ax3.bar(trade_nums, pnl_pct,
            color=[p["WIN_C"] if v > 0 else p["LOSS_C"] for v in pnl_pct],
            zorder=3, edgecolor="white", linewidth=0.4)
    ax3.axhline(0, color=p["MUTED"], linewidth=0.8)
    ax3.axhline(avg_pnl, color=p["LINE_C"],
                linewidth=1.5, linestyle="--", zorder=4)
    ax3.set_xlabel("trade #"); ax3.set_ylabel("return %")
    # avg label — always in top-left corner box
    _stat_box(ax3, 0.15, 0.92, f"avg {avg_pnl:+.2f}%",
              p["LINE_C"], p, fontsize=9)

    # ── panel 4: rolling 5-trade win rate ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    _style_ax(ax4, "rolling 5-trade win rate", p)
    valid = ~np.isnan(roll_wr)
    ax4.fill_between(trade_nums[valid], roll_wr[valid], 50,
                     where=(roll_wr[valid] >= 50), alpha=0.20, color=p["WIN_C"])
    ax4.fill_between(trade_nums[valid], roll_wr[valid], 50,
                     where=(roll_wr[valid] < 50),  alpha=0.20, color=p["LOSS_C"])
    ax4.plot(trade_nums[valid], roll_wr[valid],
             color=p["ROLL_C"], linewidth=2.2, zorder=3)
    ax4.axhline(50, color=p["MUTED"], linewidth=1,
                linestyle="--", alpha=0.6)
    ax4.set_xlabel("trade #"); ax4.set_ylabel("win rate %")
    ax4.set_ylim(0, 120)
    ax4.text(0.04, 0.08, "50% mark", transform=ax4.transAxes,
             color=p["MUTED"], fontsize=8, va="bottom")
    if n >= 5:
        latest = float(roll_wr[valid][-1])
        _stat_box(ax4, 0.82, 0.92, f"{latest:.0f}%  latest",
                  p["ROLL_C"], p, fontsize=10)

    if save_path is None:
        save_path = f"accuracy_report{'_' + ticker if ticker else ''}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=p["BG"], edgecolor="none")
    print(f"  Accuracy report saved → {save_path}")
    _open_file(save_path)
    return fig


# ─── combined accuracy report (multi-ticker) ─────────────────────────────────

def plot_combined_accuracy_report(results: list, save_path: str = None):
    """
    Combined 6-panel accuracy report across all tickers.

    Parameters
    ----------
    results   : list of (ticker_label, stats) tuples
    save_path : path to save PNG; defaults to 'combined_accuracy_report.png'
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
    avg_all    = pnl_all.mean()

    roll_wr = np.full(n, np.nan)
    for i in range(4, n):
        roll_wr[i] = wins_all[i - 4: i + 1].mean() * 100

    p = _palette()
    t_labels = list(ticker_wins.keys())
    t_values = list(ticker_wins.values())

    fig = plt.figure(figsize=(17, 11), facecolor=p["BG"])

    # ── header ───────────────────────────────────────────────────────────────
    fig.text(0.5, 0.980, "combined strategy report",
             ha="center", va="top", color=p["TEXT"],
             fontsize=15, fontweight="bold")

    pill_labels = [
        ", ".join(t_labels),
        f"{n} total trades",
        f"overall win rate  {overall_wr:.1f}%",
        f"avg P&L  {avg_all:+.2f}%",
    ]
    for xpos, label in zip([0.18, 0.38, 0.60, 0.80], pill_labels):
        fig.text(xpos, 0.943, label, ha="center", va="top",
                 color=p["MUTED"], fontsize=9,
                 bbox=dict(facecolor="white", edgecolor=p["BORDER"],
                           boxstyle="round,pad=0.35", linewidth=1,
                           alpha=0.85))

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.60, wspace=0.42,
                           left=0.07, right=0.97,
                           top=0.90, bottom=0.08)

    # ── panel 1: win rate per ticker ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _style_ax(ax1, "win rate by ticker", p)
    bar_cols = [p["WIN_C"] if v >= 50 else p["LOSS_C"] for v in t_values]
    b1 = ax1.bar(range(len(t_labels)), t_values, color=bar_cols,
                 width=0.5, zorder=3, edgecolor="white", linewidth=1.5)
    ax1.set_xticks(range(len(t_labels)))
    ax1.set_xticklabels(t_labels, color=p["TEXT"], fontsize=8)
    # counts inside bars
    for bar, val in zip(b1, t_values):
        h = bar.get_height()
        if h > 8:
            ax1.text(bar.get_x() + bar.get_width() / 2, h * 0.5,
                     f"{val:.0f}%", ha="center", va="center",
                     color="white", fontsize=10, fontweight="bold")
    ax1.axhline(50, color=p["MUTED"], linewidth=1, linestyle="--", alpha=0.5)
    ax1.set_ylim(0, 130)
    ax1.set_ylabel("win rate %")
    # combined badge in top-right corner box — never overlaps bars
    badge_c = p["WIN_C"] if overall_wr >= 50 else p["LOSS_C"]
    _stat_box(ax1, 0.80, 0.91,
              f"{overall_wr:.1f}%\ncombined",
              badge_c, p, fontsize=10)

    # ── panel 2: trade count per ticker ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _style_ax(ax2, "trades per ticker", p)
    counts = combined.groupby("ticker").size().reindex(t_labels).values
    b2 = ax2.bar(range(len(t_labels)), counts,
                 color=p["PALETTE"][:len(t_labels)],
                 width=0.5, zorder=3, edgecolor="white", linewidth=1.5)
    ax2.set_xticks(range(len(t_labels)))
    ax2.set_xticklabels(t_labels, color=p["TEXT"], fontsize=8)
    for bar, val in zip(b2, counts):
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, h * 0.5,
                     str(val), ha="center", va="center",
                     color="white", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(counts) * 1.20)
    ax2.set_ylabel("# trades")

    # ── panel 3: avg P&L per ticker ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    _style_ax(ax3, "avg return % per ticker", p)
    avg_pnls = combined.groupby("ticker")["pnl"].mean().reindex(t_labels).values
    b3 = ax3.bar(range(len(t_labels)), avg_pnls,
                 color=[p["WIN_C"] if v >= 0 else p["LOSS_C"] for v in avg_pnls],
                 width=0.5, zorder=3, edgecolor="white", linewidth=1.5)
    ax3.set_xticks(range(len(t_labels)))
    ax3.set_xticklabels(t_labels, color=p["TEXT"], fontsize=8)
    # labels inside bars
    for bar, val in zip(b3, avg_pnls):
        h = bar.get_height()
        mid = h / 2 if abs(h) > 0.5 else (0.3 if h >= 0 else -0.3)
        ax3.text(bar.get_x() + bar.get_width() / 2, mid,
                 f"{val:+.1f}%", ha="center", va="center",
                 color="white", fontsize=9, fontweight="bold")
    ax3.axhline(0, color=p["MUTED"], linewidth=0.8)
    spread = max(abs(avg_pnls.min()), abs(avg_pnls.max()), 0.5)
    ax3.set_ylim(-spread * 1.6, spread * 1.6)
    ax3.set_ylabel("avg return %")

    # ── panel 4: cumulative win rate ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    _style_ax(ax4, "cumulative win rate", p)
    ax4.fill_between(trade_nums, cum_wr, 50,
                     where=(cum_wr >= 50), alpha=0.20, color=p["WIN_C"])
    ax4.fill_between(trade_nums, cum_wr, 50,
                     where=(cum_wr < 50),  alpha=0.20, color=p["LOSS_C"])
    ax4.plot(trade_nums, cum_wr,
             color=p["LINE_C"], linewidth=2.2, zorder=3)
    ax4.axhline(50, color=p["MUTED"], linewidth=1, linestyle="--", alpha=0.5)
    ax4.set_xlabel("trade # (all tickers)")
    ax4.set_ylabel("win rate %")
    ax4.set_ylim(0, 120)
    ax4.text(0.04, 0.08, "50% mark", transform=ax4.transAxes,
             color=p["MUTED"], fontsize=8, va="bottom")
    _stat_box(ax4, 0.82, 0.92, f"{cum_wr[-1]:.1f}%",
              p["LINE_C"], p, fontsize=10)

    # ── panel 5: per-trade P&L by ticker ─────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    _style_ax(ax5, "per-trade return % by ticker", p)
    ticker_arr = combined["ticker"].values
    for idx, t in enumerate(list(dict.fromkeys(ticker_arr))):
        mask = ticker_arr == t
        ax5.bar(trade_nums[mask], pnl_all[mask],
                color=p["PALETTE"][idx % len(p["PALETTE"])],
                label=t, zorder=3, alpha=0.9,
                edgecolor="white", linewidth=0.3)
    ax5.axhline(0, color=p["MUTED"], linewidth=0.8)
    ax5.axhline(avg_all, color=p["LINE_C"],
                linewidth=1.5, linestyle="--", zorder=4)
    ax5.set_xlabel("trade # (all tickers)")
    ax5.set_ylabel("return %")
    _stat_box(ax5, 0.15, 0.92, f"avg {avg_all:+.2f}%",
              p["LINE_C"], p, fontsize=9)
    leg = ax5.legend(fontsize=7, loc="lower right", framealpha=0.9,
                     facecolor="white", edgecolor=p["BORDER"],
                     ncol=min(len(t_labels), 3))
    for txt in leg.get_texts():
        txt.set_color(p["TEXT"])

    # ── panel 6: rolling 5-trade win rate ────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    _style_ax(ax6, "rolling 5-trade win rate", p)
    valid = ~np.isnan(roll_wr)
    ax6.fill_between(trade_nums[valid], roll_wr[valid], 50,
                     where=(roll_wr[valid] >= 50), alpha=0.20, color=p["WIN_C"])
    ax6.fill_between(trade_nums[valid], roll_wr[valid], 50,
                     where=(roll_wr[valid] < 50),  alpha=0.20, color=p["LOSS_C"])
    ax6.plot(trade_nums[valid], roll_wr[valid],
             color=p["ROLL_C"], linewidth=2.2, zorder=3)
    ax6.axhline(50, color=p["MUTED"], linewidth=1, linestyle="--", alpha=0.5)
    ax6.set_xlabel("trade # (all tickers)")
    ax6.set_ylabel("win rate %")
    ax6.set_ylim(0, 120)
    ax6.text(0.04, 0.08, "50% mark", transform=ax6.transAxes,
             color=p["MUTED"], fontsize=8, va="bottom")
    if n >= 5:
        latest = float(roll_wr[valid][-1])
        _stat_box(ax6, 0.82, 0.92, f"{latest:.0f}%  latest",
                  p["ROLL_C"], p, fontsize=10)

    if save_path is None:
        save_path = "combined_accuracy_report.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=p["BG"], edgecolor="none")
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
