#!/usr/bin/env python3
"""
U.S. Equity Trading Strategy Backtesting System
美股交易策略回测系统 (v1.0)

Main Entrypoint:
Provides interactive prompts or CLI argument parsing, runs backtests,
prints a comparison summary against Buy & Hold, and generates performance charts.
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import argparse

try:
    from tabulate import tabulate
except ImportError:
    # Graceful fallback if tabulate is not installed in the active environment
    def tabulate(table_data, headers=None, tablefmt=None):
        col_widths = [len(str(h)) for h in (headers or table_data[0])]
        for row in table_data:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        lines = []
        if headers:
            lines.append("  ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers)))
            lines.append("-" * (sum(col_widths) + len(col_widths) * 2))
        for row in table_data:
            lines.append("  ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)))
        return "\n".join(lines)

from backtester.data_loader import DataLoader
from backtester.strategies.ma_crossover import DualMACrossoverStrategy
from backtester.strategies.rsi_strategy import RSIStrategy
from backtester.engine import BacktestEngine
from backtester.visualizer import Visualizer


def format_pct(val: float) -> str:
    if val is None or val != val:  # Check nan
        return "N/A"
    return f"{val * 100:+.2f}%"


def format_float(val: float, precision: int = 2) -> str:
    if val is None or val != val:
        return "N/A"
    return f"{val:.{precision}f}"


def print_comparison_table(result):
    """Prints a beautiful comparison table between Strategy and Benchmark."""
    m = result.metrics

    table_data = [
        ["Initial Capital", f"${result.initial_capital:,.2f}", f"${result.initial_capital:,.2f}"],
        ["Final Equity", f"${result.final_equity:,.2f}", f"${result.benchmark_final_equity:,.2f}"],
        ["Cumulative Return", format_pct(m["Strategy Total Return"]), format_pct(m["Benchmark Total Return"])],
        ["Annualized Return (CAGR)", format_pct(m["Strategy CAGR"]), format_pct(m["Benchmark CAGR"])],
        ["Annualized Volatility", format_pct(m["Strategy Volatility"]), format_pct(m["Benchmark Volatility"])],
        ["Sharpe Ratio (Rf=4%)", format_float(m["Strategy Sharpe"]), format_float(m["Benchmark Sharpe"])],
        ["Max Drawdown (MDD)", format_pct(m["Strategy Max Drawdown"]), format_pct(m["Benchmark Max Drawdown"])],
        ["Max DD Duration (Days)", f"{m['Max DD Duration (Days)']} days", "—"],
        ["Total Closed Trades", f"{m['Total Trades']}", "1 (Buy & Hold)"],
        ["Win Rate", format_pct(m["Win Rate"]), "—"],
        ["Profit Factor", format_float(m["Profit Factor"]) if m["Profit Factor"] != float("inf") else "∞", "—"],
    ]

    headers = ["Metric / 指标", f"Strategy ({result.strategy_name})", "Benchmark (Buy & Hold)"]

    print("\n" + "=" * 70)
    print(f"📊 BACKTEST RESULTS: {result.ticker} ({result.start_date} to {result.end_date})")
    print("=" * 70)
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
    print("=" * 70)

    # Educational beginner takeaways
    print("\n💡 策略诊断解读 (Strategy Takeaways for Beginners):")
    strat_ret = m["Strategy Total Return"]
    bench_ret = m["Benchmark Total Return"]
    strat_mdd = abs(m["Strategy Max Drawdown"])
    bench_mdd = abs(m["Benchmark Max Drawdown"])
    strat_sharpe = m["Strategy Sharpe"]
    bench_sharpe = m["Benchmark Sharpe"]

    if strat_ret > bench_ret and strat_mdd < bench_mdd:
        print("  🌟 [双赢组合]：该策略既跑赢了标的本身的买入持有收益，又控制了更小的最大回撤！")
    elif strat_mdd < bench_mdd:
        print(f"  🛡️ [风控防守]：策略最大回撤 ({strat_mdd*100:.1f}%) 显著优于买入持有 ({bench_mdd*100:.1f}%)，牺牲少许涨幅换来了更安稳的持仓体验，避免了深套。")
    elif strat_ret > bench_ret:
        print(f"  🚀 [超额收益]：策略抓住了大波段趋势，获取了更高的总回报 ({strat_ret*100:.1f}% vs {bench_ret*100:.1f}%)。")
    else:
        print("  ⚠️ [提示]：在强单边大牛市中，趋势跟踪策略可能会因震荡或中途下车而落后于买入持有，但能规避单边大熊市的暴跌。")

    if strat_sharpe > 1.0:
        print(f"  ✅ 夏普比率 ({strat_sharpe:.2f}) 处于良好区间 (>1.0)，说明每承担一份波动风险获得了较好的超额回报。")


def run_backtest_pipeline(
    ticker: str,
    strategy_type: str = "ma",
    start_date: str = None,
    end_date: str = None,
    lookback_years: int = 3,
    initial_capital: float = 10000.0,
    **kwargs,
):
    """Executes the complete backtesting pipeline."""
    ticker = ticker.strip().upper()

    # Intelligent alias for SpaceX
    if ticker in ["SPACEX", "SPACE-X", "SPACE_X"]:
        print("\n🚀 提示：SpaceX 已在纳斯达克挂牌上市，股票代码为 SPCX。已自动为您切换为代码 SPCX 进行回测分析！\n")
        ticker = "SPCX"

    loader = DataLoader()
    try:
        data = loader.load_data(ticker, start_date=start_date, end_date=end_date, lookback_years=lookback_years)
    except Exception as e:
        print(f"\n❌ 获取数据失败: {e}")
        return None

    # Instantiate strategy
    if strategy_type.lower() in ["ma", "dma", "moving_average"]:
        short_w = kwargs.get("short_window", 20)
        long_w = kwargs.get("long_window", 50)
        ma_type = kwargs.get("ma_type", "SMA")
        strategy = DualMACrossoverStrategy(short_window=short_w, long_window=long_w, ma_type=ma_type)
    elif strategy_type.lower() in ["rsi", "mean_reversion"]:
        period = kwargs.get("rsi_period", 14)
        oversold = kwargs.get("oversold", 30.0)
        overbought = kwargs.get("overbought", 70.0)
        use_trend = kwargs.get("use_trend_filter", True)
        strategy = RSIStrategy(period=period, oversold=oversold, overbought=overbought, use_trend_filter=use_trend)
    else:
        print(f"❌ 未知策略类型: {strategy_type}")
        return None

    # Run backtest
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_per_trade=kwargs.get("commission", 0.0),
        slippage_bps=kwargs.get("slippage_bps", 5.0),
    )
    result = engine.run(data, strategy, ticker=ticker)

    # Print summary
    print_comparison_table(result)

    # Plot chart
    chart_path = Visualizer.plot_result(result)
    print(f"\n📈 图表生成完毕！您可以在此处查看完整可视化收益曲线：\n👉 {chart_path}\n")

    return result


def interactive_menu():
    """Provides a beginner-friendly interactive terminal menu."""
    print("=" * 70)
    print(" 🚀 美股交易策略回测系统 (U.S. Equity Trading Backtesting System) v1.0")
    print("=" * 70)

    # 1. Stock selection
    print("\n[步骤 1/4] 请输入你想测试的美股代码（例如：SPCX, TSLA, NVDA, AAPL）")
    ticker_input = input("股票代码 [默认 NVDA]: ").strip().upper()
    ticker = ticker_input if ticker_input else "NVDA"

    # 2. Strategy selection
    print("\n[步骤 2/4] 请选择交易策略：")
    print("  1. 双均线交叉策略 (Dual MA Crossover: 20 SMA 金叉 50 SMA 买入，死叉卖出)")
    print("  2. RSI 动量回归策略 (RSI Mean Reversion: RSI<30 超卖逢低买入，RSI>70 超买止盈)")
    choice = input("请输入选项编号 (1 或 2) [默认 1]: ").strip()

    if choice == "2":
        strategy_type = "rsi"
    else:
        strategy_type = "ma"

    # 3. Timeframe
    print("\n[步骤 3/4] 请选择回测时间跨度：")
    print("  1. 最近 3 年 (推荐)")
    print("  2. 最近 5 年")
    print("  3. 最近 1 年")
    time_choice = input("请输入选项编号 [默认 1]: ").strip()
    years_map = {"1": 3, "2": 5, "3": 1}
    lookback = years_map.get(time_choice, 3)

    # 4. Capital
    print("\n[步骤 4/4] 初始模拟资金 (USD) [默认 10,000]: ")
    capital_input = input("资金金额: ").strip()
    try:
        capital = float(capital_input) if capital_input else 10000.0
    except ValueError:
        capital = 10000.0

    print("\n🚀 开始启动回测引擎...\n")
    run_backtest_pipeline(
        ticker=ticker,
        strategy_type=strategy_type,
        lookback_years=lookback,
        initial_capital=capital,
    )


def main():
    parser = argparse.ArgumentParser(description="U.S. Equity Trading Strategy Backtesting System")
    parser.add_argument("--ticker", "-t", type=str, help="Stock symbol (e.g. NVDA, TSLA, AAPL)")
    parser.add_argument("--strategy", "-s", type=str, choices=["ma", "rsi"], default="ma", help="Strategy type")
    parser.add_argument("--years", "-y", type=int, default=3, help="Lookback years (default: 3)")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", "-c", type=float, default=10000.0, help="Initial capital (default: 10000)")
    parser.add_argument("--short-ma", type=int, default=20, help="Short MA window (default: 20)")
    parser.add_argument("--long-ma", type=int, default=50, help="Long MA window (default: 50)")

    args = parser.parse_args()

    # If no ticker passed via CLI, launch interactive wizard
    if not args.ticker:
        interactive_menu()
    else:
        run_backtest_pipeline(
            ticker=args.ticker,
            strategy_type=args.strategy,
            start_date=args.start,
            end_date=args.end,
            lookback_years=args.years,
            initial_capital=args.capital,
            short_window=args.short_ma,
            long_window=args.long_ma,
        )


if __name__ == "__main__":
    main()
