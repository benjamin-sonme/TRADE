"""
Portfolio Visualizer Module
Plots multi-asset strategy equity curve, dynamic allocation stacked area chart,
and underwater drawdown comparisons.
"""

import io
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from .portfolio_engine import PortfolioBacktestResult

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


class PortfolioVisualizer:
    """Visualizes multi-asset portfolio backtests."""

    @staticmethod
    def build_figure(result: PortfolioBacktestResult):
        """Builds a 3-panel matplotlib figure for portfolio backtest results."""
        data = result.equity_df
        metrics = result.metrics
        weights = result.weights_history

        fig, (ax_equity, ax_alloc, ax_dd) = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(14, 12),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1.6, 1]},
        )
        fig.patch.set_facecolor("#0F172A")  # Dark slate background

        for ax in [ax_equity, ax_alloc, ax_dd]:
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
            label=f"{result.strategy_name} ({strat_final_pct:+.1f}%)",
            color="#10B981",  # Emerald Green
            linewidth=2.2,
        )
        ax_equity.plot(
            data.index,
            data["Benchmark_Equity"],
            label=f"{result.benchmark_name} ({bench_final_pct:+.1f}%)",
            color="#38BDF8",  # Sky Blue
            linewidth=1.8,
            linestyle="--",
        )
        ax_equity.axhline(result.initial_capital, color="#64748B", linestyle=":", alpha=0.7, label="Initial Capital")
        ax_equity.set_title(
            f"Portfolio Backtest | {result.strategy_name} vs {result.benchmark_name} ({result.start_date} to {result.end_date})",
            color="#F8FAFC",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )
        ax_equity.set_ylabel("Portfolio Value ($)", color="#F8FAFC", fontsize=11)
        ax_equity.legend(loc="upper left", facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC")

        # ---------------- 2. Asset Allocation Over Time ----------------
        # Distinct palette for assets
        palette = [
            "#38BDF8", "#F59E0B", "#EC4899", "#8B5CF6", "#10B981",
            "#F97316", "#06B6D4", "#EAB308", "#A855F7", "#64748B"
        ]
        
        # Sort columns so CASH is at the bottom or top
        cols = [c for c in weights.columns if c != "CASH"]
        if "CASH" in weights.columns:
            cols.append("CASH")

        y_stack = [weights[c].values * 100 for c in cols]
        labels = [c for c in cols]
        colors = [palette[idx % len(palette)] if c != "CASH" else "#475569" for idx, c in enumerate(cols)]

        ax_alloc.stackplot(weights.index, y_stack, labels=labels, colors=colors, alpha=0.85)
        ax_alloc.set_ylabel("Allocation Weight (%)", color="#F8FAFC", fontsize=11)
        ax_alloc.set_ylim(0, 100)
        ax_alloc.legend(loc="upper left", facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC", ncol=min(6, len(cols)), fontsize=9)

        # ---------------- 3. Underwater Drawdowns ----------------
        strat_dd = metrics["Drawdown Series"] * 100
        bench_dd = metrics["Benchmark Drawdown Series"] * 100

        ax_dd.plot(data.index, strat_dd, label=f"Strategy DD (Max: {metrics['Strategy Max Drawdown']*100:.1f}%)", color="#F43F5E", linewidth=1.4)
        ax_dd.plot(data.index, bench_dd, label=f"Benchmark DD (Max: {metrics['Benchmark Max Drawdown']*100:.1f}%)", color="#64748B", linewidth=1.0, linestyle=":")
        ax_dd.fill_between(data.index, strat_dd, 0, color="#F43F5E", alpha=0.25)
        ax_dd.set_ylabel("Drawdown (%)", color="#F8FAFC", fontsize=11)
        ax_dd.set_xlabel("Trading Date", color="#F8FAFC", fontsize=11)
        ax_dd.legend(loc="lower left", facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC")

        date_fmt = "%Y-%m-%d" if len(data) <= 90 else "%Y-%m"
        ax_dd.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
        fig.autofmt_xdate()

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_result(result: PortfolioBacktestResult, save_path: str = None) -> str:
        """Saves portfolio chart to disk and returns the saved file path."""
        fig = PortfolioVisualizer.build_figure(result)

        if save_path is None:
            clean_strat = result.strategy_name.replace(" ", "_")
            save_path = os.path.join(REPORTS_DIR, f"{clean_strat}_{result.start_date}_{result.end_date}.png")

        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        print(f"[PortfolioVisualizer] Report chart saved to: {save_path}")
        return save_path

    @staticmethod
    def render_png_bytes(result: PortfolioBacktestResult) -> bytes:
        """Renders portfolio chart into in-memory PNG bytes without saving to disk."""
        fig = PortfolioVisualizer.build_figure(result)
        buf = io.BytesIO()
        fig.savefig(buf, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
