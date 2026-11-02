"""
Visualizer Module
Generates high-resolution performance charts comparing Strategy vs Buy & Hold Benchmark,
including Equity Curves, Price Action with Trade Markers, and Underwater Drawdowns.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from .engine import BacktestResult

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


class Visualizer:
    """Creates clear comparative charts for backtesting results."""

    @staticmethod
    def plot_result(result: BacktestResult, save_path: str = None) -> str:
        """
        Plots a 3-panel chart:
        1. Equity Curve: Strategy vs Buy & Hold
        2. Price Chart with Indicator & Trade Entry/Exit Markers
        3. Underwater Drawdown Curve
        """
        data = result.equity_df
        metrics = result.metrics

        fig, (ax_equity, ax_price, ax_dd) = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(14, 11),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1.5, 1]},
        )
        fig.patch.set_facecolor("#0F172A")  # Dark slate background

        for ax in [ax_equity, ax_price, ax_dd]:
            ax.set_facecolor("#1E293B")
            ax.tick_params(colors="#94A3B8", labelsize=10)
            ax.spines["bottom"].set_color("#334155")
            ax.spines["top"].set_color("#334155")
            ax.spines["left"].set_color("#334155")
            ax.spines["right"].set_color("#334155")
            ax.grid(True, linestyle="--", alpha=0.2, color="#94A3B8")

        # ---------------- 1. Equity Curve ----------------
        strat_final_pct = metrics["Strategy Total Return"] * 100
        bench_final_pct = metrics["Benchmark Total Return"] * 100

        ax_equity.plot(
            data.index,
            data["Strategy_Equity"],
            label=f"Strategy ({strat_final_pct:+.1f}%)",
            color="#10B981",  # Emerald Green
            linewidth=2.0,
        )
        ax_equity.plot(
            data.index,
            data["Benchmark_Equity"],
            label=f"Buy & Hold ({bench_final_pct:+.1f}%)",
            color="#38BDF8",  # Sky Blue
            linewidth=1.8,
            linestyle="--",
        )
        ax_equity.axhline(result.initial_capital, color="#64748B", linestyle=":", alpha=0.7, label="Initial Capital")
        ax_equity.set_title(
            f"{result.ticker} Backtest | {result.strategy_name} vs Buy & Hold ({result.start_date} to {result.end_date})",
            color="#F8FAFC",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )
        ax_equity.set_ylabel("Portfolio Value ($)", color="#F8FAFC", fontsize=11)
        ax_equity.legend(loc="upper left", facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC")

        # ---------------- 2. Price Chart & Trade Markers ----------------
        ax_price.plot(data.index, data["Close"], color="#CBD5E1", linewidth=1.2, label=f"{result.ticker} Close")

        # Indicators if present
        if "MA_Short" in data.columns and "MA_Long" in data.columns:
            ax_price.plot(data.index, data["MA_Short"], color="#F59E0B", linewidth=1.0, alpha=0.9, label="MA Short")
            ax_price.plot(data.index, data["MA_Long"], color="#8B5CF6", linewidth=1.0, alpha=0.9, label="MA Long")
        elif "Upper_Channel" in data.columns and "Lower_Channel" in data.columns:
            ax_price.plot(data.index, data["Upper_Channel"], color="#10B981", linewidth=1.0, linestyle="--", label="Upper Donchian")
            ax_price.plot(data.index, data["Lower_Channel"], color="#EF4444", linewidth=1.0, linestyle="--", label="Lower Donchian")
        elif "BB_Upper" in data.columns and "BB_Lower" in data.columns:
            ax_price.plot(data.index, data["BB_Upper"], color="#38BDF8", linewidth=1.0, linestyle="--", label="BB Upper")
            ax_price.plot(data.index, data["BB_Middle"], color="#F59E0B", linewidth=1.0, label="BB 20 SMA")
            ax_price.plot(data.index, data["BB_Lower"], color="#38BDF8", linewidth=1.0, linestyle="--", label="BB Lower")
            ax_price.fill_between(data.index, data["BB_Lower"], data["BB_Upper"], color="#38BDF8", alpha=0.08)
        elif "SMA_Fast" in data.columns and "SMA_Mid" in data.columns and "SMA_Slow" in data.columns:
            ax_price.plot(data.index, data["SMA_Fast"], color="#38BDF8", linewidth=1.0, label="SMA 20")
            ax_price.plot(data.index, data["SMA_Mid"], color="#F59E0B", linewidth=1.0, label="SMA 50")
            ax_price.plot(data.index, data["SMA_Slow"], color="#A855F7", linewidth=1.0, label="SMA 200")
        elif "Trend_MA" in data.columns:
            ax_price.plot(data.index, data["Trend_MA"], color="#8B5CF6", linewidth=1.0, alpha=0.9, label="Trend MA")

        # Plot Buy / Sell trade markers
        buy_trades = [t for t in result.trades if t.position_type == "LONG"]
        if buy_trades:
            buy_dates = [t.entry_date for t in buy_trades]
            buy_prices = [t.entry_price for t in buy_trades]
            ax_price.scatter(
                buy_dates,
                buy_prices,
                marker="^",
                color="#22C55E",
                s=80,
                zorder=5,
                label=f"Buy Signal ({len(buy_trades)})",
            )

            sell_dates = [t.exit_date for t in buy_trades]
            sell_prices = [t.exit_price for t in buy_trades]
            ax_price.scatter(
                sell_dates,
                sell_prices,
                marker="v",
                color="#EF4444",
                s=80,
                zorder=5,
                label=f"Sell Signal ({len(buy_trades)})",
            )

        ax_price.set_ylabel("Stock Price ($)", color="#F8FAFC", fontsize=11)
        ax_price.legend(loc="upper left", facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC")

        # ---------------- 3. Underwater Drawdowns ----------------
        strat_dd = metrics["Drawdown Series"] * 100
        bench_dd = metrics["Benchmark Drawdown Series"] * 100

        ax_dd.plot(data.index, strat_dd, label=f"Strategy DD (Max: {metrics['Strategy Max Drawdown']*100:.1f}%)", color="#F43F5E", linewidth=1.4)
        ax_dd.plot(data.index, bench_dd, label=f"Benchmark DD (Max: {metrics['Benchmark Max Drawdown']*100:.1f}%)", color="#64748B", linewidth=1.0, linestyle=":")
        ax_dd.fill_between(data.index, strat_dd, 0, color="#F43F5E", alpha=0.25)
        ax_dd.set_ylabel("Drawdown (%)", color="#F8FAFC", fontsize=11)
        ax_dd.legend(loc="lower left", facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC")

        # Formatting x-axis dates
        ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()

        plt.tight_layout()

        if save_path is None:
            clean_strat = result.strategy_name.replace(" ", "_")
            save_path = os.path.join(REPORTS_DIR, f"{result.ticker}_{clean_strat}_{result.start_date}_{result.end_date}.png")

        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        print(f"[Visualizer] Report chart saved to: {save_path}")
        return save_path
