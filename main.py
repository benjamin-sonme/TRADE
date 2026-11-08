#!/usr/bin/env python3
"""
U.S. Equity Trading Strategy Backtesting System
美股交易策略回测系统 (v2.1)

Supports 9 Classic Trading & Portfolio Strategies:
S01. Dual MA Crossover (双均线交叉)
S02. RSI Mean Reversion (RSI 均值回归)
S03. Donchian Channel Breakout (唐奇安通道突破)
S04. Relative Momentum Rotation (相对动量轮动)
S05. Bollinger Band Squeeze (布林带挤压突破)
S06. Trend Pullback (趋势回撤买入)
S07. Threshold Rebalancing (阈值再平衡)
S08. Quality & Value Factor (质量与价值因子)
S09. Dual Momentum (双重动量轮动)
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import argparse
from typing import Dict, Any, List, Optional

try:
    from tabulate import tabulate
except ImportError:
    def tabulate(table_data, headers=None, tablefmt=None):
        col_widths = [len(str(h)) for h in (headers or table_data[0])]
        for row in table_data:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        lines = []
        if headers:
            lines.extend(
                (
                    "  ".join(
                        str(h).ljust(col_widths[i])
                        for i, h in enumerate(headers)
                    ),
                    "-" * (sum(col_widths) + len(col_widths) * 2),
                )
            )
        lines.extend(
            "  ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
            for row in table_data
        )
        return "\n".join(lines)

from backtester.data_loader import DataLoader
from backtester.engine import BacktestEngine
from backtester.visualizer import Visualizer
from backtester.portfolio_engine import PortfolioEngine, PortfolioBacktestResult
from backtester.portfolio_visualizer import PortfolioVisualizer
from backtester.risk_analyzer import assess_investment_risk, print_risk_assessment_card
from backtester.universe import NASDAQ_TOP_50_TICKERS, NASDAQ_TOP_50_MAP, print_nasdaq_top50_list

# Single stock strategies
from backtester.strategies.ma_crossover import DualMACrossoverStrategy
from backtester.strategies.rsi_strategy import RSIStrategy
from backtester.strategies.donchian_breakout import DonchianBreakoutStrategy
from backtester.strategies.bollinger_squeeze import BollingerBandSqueezeStrategy
from backtester.strategies.trend_pullback import TrendPullbackStrategy

# Portfolio / Multi-asset strategies
from backtester.strategies.relative_momentum import RelativeMomentumRotation
from backtester.strategies.threshold_rebalancing import ThresholdRebalancing
from backtester.strategies.quality_value_factor import QualityValueFactorStrategy, DEFAULT_FACTOR_UNIVERSE
from backtester.strategies.dual_momentum import DualMomentumStrategy


TIMEFRAME_PRESETS = {
    "1": {"code": "3y", "label": "最近 3 年 (推荐)", "eval_days": 756, "years": 3},
    "2": {"code": "5y", "label": "最近 5 年", "eval_days": 1260, "years": 5},
    "3": {"code": "1y", "label": "最近 1 年", "eval_days": 252, "years": 1},
    "4": {"code": "6m", "label": "最近半年 (6个月)", "eval_days": 126, "years": 1},
    "5": {"code": "1m", "label": "最近 1 个月", "eval_days": 21, "years": 1},
    "6": {"code": "1w", "label": "最近 1 周 (5个交易日)", "eval_days": 5, "years": 1},
    "7": {"code": "3d", "label": "最近 3 天 (3个交易日)", "eval_days": 3, "years": 1},
}

TIMEFRAME_CODE_MAP = {
    cfg["code"]: cfg for cfg in TIMEFRAME_PRESETS.values()
}


def format_pct(val: float) -> str:
    return "N/A" if (val is None or val != val) else f"{val * 100:+.2f}%"


def format_float(val: float, precision: int = 2) -> str:
    return "N/A" if (val is None or val != val) else f"{val:.{precision}f}"


def print_comparison_table(result, is_portfolio: bool = False):
    """Prints a beautiful comparison table and investment risk assessment card."""
    m = result.metrics

    strat_header = f"Strategy ({result.strategy_name})"
    bench_header = f"Benchmark ({getattr(result, 'benchmark_name', 'Buy & Hold')})"

    table_data = [
        ["Initial Capital", f"${result.initial_capital:,.2f}", f"${result.initial_capital:,.2f}"],
        ["Final Equity", f"${result.final_equity:,.2f}", f"${result.benchmark_final_equity:,.2f}"],
        ["Cumulative Return", format_pct(m["Strategy Total Return"]), format_pct(m["Benchmark Total Return"])],
        ["Annualized Return (CAGR)", format_pct(m["Strategy CAGR"]), format_pct(m["Benchmark CAGR"])],
        ["Annualized Volatility", format_pct(m["Strategy Volatility"]), format_pct(m["Benchmark Volatility"])],
        ["Sharpe Ratio (Rf=4%)", format_float(m["Strategy Sharpe"]), format_float(m["Benchmark Sharpe"])],
        ["Max Drawdown (MDD)", format_pct(m["Strategy Max Drawdown"]), format_pct(m["Benchmark Max Drawdown"])],
        ["Max DD Duration (Days)", f"{m.get('Max DD Duration (Days)', '—')} days", "—"],
    ]

    if not is_portfolio:
        table_data.extend([
            ["Total Closed Trades", f"{m.get('Total Trades', 0)}", "1 (Buy & Hold)"],
            ["Win Rate", format_pct(m.get("Win Rate", 0)), "—"],
            ["Profit Factor", format_float(m.get("Profit Factor", 0)) if m.get("Profit Factor", 0) != float("inf") else "∞", "—"],
        ])
    else:
        events = getattr(result, "rebalance_events", [])
        table_data.append(["Rebalance Events", f"{len(events)} times", "—"])

    headers = ["Metric / 指标", strat_header, bench_header]

    title = getattr(result, "ticker", result.strategy_name)
    print("\n" + "=" * 75)
    print(f"📊 BACKTEST RESULTS: {title} ({result.start_date} to {result.end_date})")
    print("=" * 75)
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
    print("=" * 75)

    # 1. Strategy Diagnostic Takeaways
    print("\n💡 策略诊断解读 (Strategy Takeaways for Beginners):")
    strat_ret = m["Strategy Total Return"]
    bench_ret = m["Benchmark Total Return"]
    strat_mdd = abs(m["Strategy Max Drawdown"])
    bench_mdd = abs(m["Benchmark Max Drawdown"])
    strat_sharpe = m["Strategy Sharpe"]

    if strat_ret > bench_ret and strat_mdd < bench_mdd:
        print("  🌟 [双赢组合]：该策略既跑赢了标的本身的买入持有收益，又控制了更小的最大回撤！")
    elif strat_mdd < bench_mdd:
        print(f"  🛡️ [风控防守]：策略最大回撤 ({strat_mdd*100:.1f}%) 显著优于基准 ({bench_mdd*100:.1f}%)，有效避免了深套和踩雷。")
    elif strat_ret > bench_ret:
        print(f"  🚀 [超额收益]：策略捕捉到了更佳的收益波段，获得了更高的总回报 ({strat_ret*100:.1f}% vs {bench_ret*100:.1f}%)。")
    else:
        print("  ⚠️ [提示]：在强单边大牛市中，规则策略可能因等待确认信号或防守空仓而落后于买入持有，但熊市防御价值显著。")

    if strat_sharpe > 1.0:
        print(f"  ✅ 夏普比率 ({strat_sharpe:.2f}) 处于优异区间 (>1.0)，风险调整后超额收益出色。")

    # 2. Comprehensive Investment Risk Rating Card
    risk_profile = assess_investment_risk(m, ticker=title)
    print_risk_assessment_card(risk_profile)


def handle_chart_generation(
    result,
    is_portfolio: bool = False,
    auto_chart: Optional[bool] = None,
):
    """
    Asks the user for confirmation before generating and saving visualization charts.
    """
    if auto_chart is True:
        proceed = True
    elif auto_chart is False:
        proceed = False
    else:
        # Prompt user interactively
        ans = input("📊 是否生成并保存高清可视化分析图表至 reports/ 目录？(y/n) [默认 y]: ").strip().lower()
        proceed = ans in {"", "y", "yes"}

    if proceed:
        print("🎨 正在渲染高清可视化分析图表，请稍候...")
        if is_portfolio:
            chart_path = PortfolioVisualizer.plot_result(result)
        else:
            chart_path = Visualizer.plot_result(result)
        print(f"📈 图表生成完毕！您可以在此处查看完整走势图：\n👉 {chart_path}\n")
    else:
        print("\n⏩ [已跳过图表生成]\n")


def run_single_stock_pipeline(
    ticker: str,
    strategy_code: str,
    lookback_years: int = 3,
    eval_days: Optional[int] = None,
    initial_capital: float = 10000.0,
    auto_chart: Optional[bool] = None,
    **kwargs,
):
    """Executes single stock strategy backtest (S01, S02, S03, S05, S06)."""
    ticker = ticker.strip().upper()
    if ticker in {"SPACEX", "SPACE-X", "SPACE_X"}:
        print("\n🚀 提示：SpaceX 已在纳斯达克挂牌上市，股票代码为 SPCX。已自动为您切换为代码 SPCX 进行回测分析！\n")
        ticker = "SPCX"

    loader = DataLoader()
    try:
        data = loader.load_data(ticker, lookback_years=lookback_years)
    except Exception as e:
        print(f"\n❌ 获取数据失败: {e}")
        return None

    code = strategy_code.upper()
    if code in {"S01", "MA"}:
        strategy = DualMACrossoverStrategy(
            short_window=kwargs.get("short_ma", 20),
            long_window=kwargs.get("long_ma", 50),
        )
    elif code in {"S02", "RSI"}:
        strategy = RSIStrategy(period=14, oversold=30.0, overbought=70.0, use_trend_filter=True)
    elif code in {"S03", "DONCHIAN"}:
        strategy = DonchianBreakoutStrategy(
            entry_window=kwargs.get("entry_window", 20),
            exit_window=kwargs.get("exit_window", 10),
        )
    elif code in {"S05", "BOLLINGER", "BB"}:
        strategy = BollingerBandSqueezeStrategy(period=20, std_dev=2.0, squeeze_percentile=10.0)
    elif code in {"S06", "PULLBACK"}:
        strategy = TrendPullbackStrategy(sma_fast=20, sma_mid=50, sma_slow=200)
    else:
        print(f"❌ 未知单股策略代码: {strategy_code}")
        return None

    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_per_trade=kwargs.get("commission", 0.0),
        slippage_bps=kwargs.get("slippage_bps", 5.0),
    )
    result = engine.run(data, strategy, ticker=ticker, eval_days=eval_days)

    print_comparison_table(result, is_portfolio=False)
    handle_chart_generation(result, is_portfolio=False, auto_chart=auto_chart)
    return result


def run_portfolio_pipeline(
    strategy_code: str,
    lookback_years: int = 3,
    eval_days: Optional[int] = None,
    initial_capital: float = 100000.0,
    auto_chart: Optional[bool] = None,
    **kwargs,
):
    """Executes multi-asset / portfolio strategy backtest (S04, S07, S08, S09)."""
    loader = DataLoader()
    p_engine = PortfolioEngine(initial_capital=initial_capital)
    code = strategy_code.upper()

    if code in {"S09", "DUAL_MOMENTUM"}:
        # S09: Dual Momentum (SPY, VEU, BIL)
        print("\n🚀 [S09] 加载宏观核心资产: SPY (美股), VEU (全球除美), BIL (1-3月短期国债/现金)...")
        prices_df = loader.load_multi_data(["SPY", "VEU", "BIL"], lookback_years=lookback_years)
        dm = DualMomentumStrategy()
        weights_df = dm.compute_target_weights(prices_df)
        result = p_engine.run(
            prices_df=prices_df,
            target_weights_df=weights_df,
            strategy_name="S09_DualMomentum",
            benchmark_series=prices_df["SPY"],
            benchmark_name="SPY (US Equities)",
            eval_days=eval_days,
        )

    elif code in {"S04", "RELATIVE_MOMENTUM"}:
        # S04: Relative Momentum Rotation
        universe = kwargs.get("universe") or ["NVDA", "TSLA", "SPCX", "AAPL", "MSFT", "AMZN", "GOOGL", "META"]
        print(f"\n🚀 [S04] 加载候选股票池 ({len(universe)}只标的): {', '.join(universe[:10])}{' ...' if len(universe) > 10 else ''}...")
        prices_df = loader.load_multi_data(universe, lookback_years=lookback_years)
        rm = RelativeMomentumRotation(top_n=4, target_weight_per_stock=0.20)
        weights_df = rm.compute_target_weights(prices_df)
        
        # Benchmark: Equal-weight buy & hold of the universe
        norm_prices = prices_df.divide(prices_df.iloc[0])
        benchmark_series = norm_prices.mean(axis=1) * prices_df.iloc[0].mean()
        result = p_engine.run(
            prices_df=prices_df,
            target_weights_df=weights_df,
            strategy_name="S04_RelativeMomentumRotation",
            benchmark_series=benchmark_series,
            benchmark_name="Equal-Weight Buy & Hold",
            eval_days=eval_days,
        )

    elif code in {"S07", "REBALANCE"}:
        # S07: Threshold Rebalancing
        universe = kwargs.get("universe") or ["NVDA", "TSLA", "SPCX", "AAPL"]
        print(f"\n🚀 [S07] 加载组合持仓 ({len(universe)}只核心标的): {', '.join(universe[:10])}...")
        prices_df = loader.load_multi_data(universe, lookback_years=lookback_years)
        
        tw_per_stock = 0.80 / len(universe)
        target_weights = {asset: tw_per_stock for asset in universe}
        tr = ThresholdRebalancing(target_weights=target_weights, target_cash_pct=0.20, band_pct=0.25)
        
        result = tr.simulate(
            prices_df=prices_df,
            initial_capital=initial_capital,
            benchmark_name="Buy & Hold (No Rebalance)",
            eval_days=eval_days,
        )

    elif code in {"S08", "FACTOR", "QUALITY_VALUE"}:
        # S08: Quality & Value Factor
        universe = kwargs.get("universe") or DEFAULT_FACTOR_UNIVERSE
        print(f"\n🚀 [S08] 加载基本面多因子候选池 ({len(universe)}只标的)...")
        prices_df = loader.load_multi_data(universe, lookback_years=lookback_years)
        qf = QualityValueFactorStrategy(universe=universe, top_n=5, target_weight_per_stock=0.15)
        weights_df = qf.compute_target_weights(prices_df)
        
        # Benchmark: Equal-weight buy & hold of the universe
        norm_prices = prices_df.divide(prices_df.iloc[0])
        benchmark_series = norm_prices.mean(axis=1) * prices_df.iloc[0].mean()
        result = p_engine.run(
            prices_df=prices_df,
            target_weights_df=weights_df,
            strategy_name="S08_QualityValueFactor",
            benchmark_series=benchmark_series,
            benchmark_name="Universe Equal-Weight",
            eval_days=eval_days,
        )

    else:
        print(f"❌ 未知组合策略代码: {strategy_code}")
        return None

    print_comparison_table(result, is_portfolio=True)
    handle_chart_generation(result, is_portfolio=True, auto_chart=auto_chart)
    return result


MENU_TEXT = """===========================================================================
 🚀 美股交易策略回测系统 (U.S. Equity Trading Backtesting System) v2.1
===========================================================================
请选择你想测试的交易策略体系：
-----------------------------------------------------------------------
【单股票技术指标策略 (Single-Stock Technical Strategies)】
  1. [S01] 双均线交叉策略 (Dual MA Crossover - 趋势跟踪)
  2. [S02] RSI 均值回归策略 (RSI Mean Reversion - 超卖低吸)
  3. [S03] 唐奇安通道突破策略 (Donchian Channel Breakout - 突破交易)
  5. [S05] 布林带挤压突破策略 (Bollinger Band Squeeze - 波动率突破)
  6. [S06] 趋势回撤买入策略 (Trend Pullback - 顺大势逆小势)

【多资产组合与宏观轮动策略 (Portfolio & Macro Rotation Strategies)】
  4. [S04] 相对动量轮动策略 (Relative Momentum Rotation - 强中选强)
  7. [S07] 阈值再平衡策略 (Threshold Rebalancing - 组合风险控制)
  8. [S08] 质量与价值因子策略 (Quality & Value Factor - 财报基本面选股)
  9. [S09] 双重动量策略 (Dual Momentum - 全球股债防御性轮动)
-----------------------------------------------------------------------"""


def select_portfolio_universe() -> List[str]:
    """Interactive prompt to select asset universe for portfolio strategies."""
    print("\n[资产池选择 (Universe Selection)]")
    print("  1. 核心精选科技资产池 (Mag 7: NVDA, TSLA, SPCX, AAPL, MSFT, AMZN, GOOGL, META)")
    print(f"  2. 纳斯达克市值前 50 强标的池 (Nasdaq Top 50: 共 {len(NASDAQ_TOP_50_TICKERS)} 家龙头)")
    print("  3. 查看纳斯达克 Top 50 详细名单")
    pool_choice = input("请选择资产池 (1/2/3) [默认 1]: ").strip() or "1"

    if pool_choice == "3":
        print_nasdaq_top50_list()
        pool_choice = input("请选择资产池 (1: 核心科技池, 2: 纳斯达克前50) [默认 1]: ").strip() or "1"

    return NASDAQ_TOP_50_TICKERS if pool_choice == "2" else ["NVDA", "TSLA", "SPCX", "AAPL", "MSFT", "AMZN", "GOOGL", "META"]


def interactive_menu():
    """Interactive CLI menu for selecting from all 9 strategies."""
    print(MENU_TEXT)
    choice = input("请输入策略编号 (1-9) [默认 1]: ").strip() or "1"

    choice_map = {
        "1": "S01",
        "2": "S02",
        "3": "S03",
        "4": "S04",
        "5": "S05",
        "6": "S06",
        "7": "S07",
        "8": "S08",
        "9": "S09",
    }
    selected_code = choice_map.get(choice, "S01")

    # Timeframe selection (Expanded to 7 tiers!)
    print("\n[回测时间跨度]")
    for key, val in TIMEFRAME_PRESETS.items():
        print(f"  {key}. {val['label']}")
    time_choice = input("请输入时间档位编号 (1-7) [默认 1]: ").strip() or "1"
    tf_info = TIMEFRAME_PRESETS.get(time_choice, TIMEFRAME_PRESETS["1"])
    eval_days = tf_info["eval_days"]
    years = tf_info["years"]

    if selected_code in {"S01", "S02", "S03", "S05", "S06"}:
        print("\n[股票代码选择]")
        print("💡 提示：输入美股代码即可；输入 '?' 可查看纳斯达克市值前50强公司名单。")
        while True:
            ticker_input = input("请输入测试的美股代码（如 NVDA, TSLA, SPCX）[默认 NVDA]: ").strip().upper()
            if ticker_input in {"?", "LIST", "TOP50"}:
                print_nasdaq_top50_list()
                continue
            ticker = ticker_input or "NVDA"
            break

        run_single_stock_pipeline(
            ticker=ticker,
            strategy_code=selected_code,
            lookback_years=years,
            eval_days=eval_days,
        )
    else:
        run_portfolio_pipeline(
            strategy_code=selected_code,
            lookback_years=years,
            eval_days=eval_days,
            universe=select_portfolio_universe(),
        )


def main():
    parser = argparse.ArgumentParser(description="U.S. Equity Trading Strategy Backtesting System v2.1")
    parser.add_argument("--strategy", "-s", type=str, help="Strategy code (s01-s09 or ma, rsi, donchian, momentum, bollinger, pullback, rebalance, factor, dual_momentum)")
    parser.add_argument("--ticker", "-t", type=str, default=None, help="Stock symbol for single-stock strategies (e.g. NVDA, TSLA, SPCX, or 'top50')")
    parser.add_argument("--timeframe", "-p", type=str, choices=["5y", "3y", "1y", "6m", "1m", "1w", "3d"], default=None, help="Timeframe (5y, 3y, 1y, 6m, 1m, 1w, 3d)")
    parser.add_argument("--years", "-y", type=int, default=3, help="Lookback years (default: 3)")
    parser.add_argument("--capital", "-c", type=float, default=None, help="Initial capital (default: 10000 for single, 100000 for portfolio)")
    parser.add_argument("--nasdaq-top50", action="store_true", help="Use Nasdaq Top 50 universe")
    parser.add_argument("--chart", action="store_true", help="Auto generate charts without prompting")
    parser.add_argument("--no-chart", action="store_true", help="Skip charts without prompting")
    parser.add_argument("--ui", action="store_true", help="Launch Streamlit Web UI")

    args = parser.parse_args()

    if args.ui:
        import subprocess
        streamlit_bin = os.path.join(os.path.dirname(sys.executable), "streamlit")
        cmd = [streamlit_bin if os.path.exists(streamlit_bin) else "streamlit", "run", "app.py"]
        print(f"🚀 正在启动 Web 控制台: {' '.join(cmd)}")
        subprocess.run(cmd)
        return

    # Determine auto_chart preference
    auto_chart = None
    if args.chart:
        auto_chart = True
    elif args.no_chart:
        auto_chart = False

    # Determine evaluation days from timeframe argument
    eval_days = None
    years = args.years
    if args.timeframe and (tf_cfg := TIMEFRAME_CODE_MAP.get(args.timeframe.lower())):
        eval_days = tf_cfg["eval_days"]
        years = tf_cfg["years"]

    if not args.strategy:
        interactive_menu()
    else:
        strat = args.strategy.lower()
        single_strats = {"s01", "ma", "s02", "rsi", "s03", "donchian", "s05", "bollinger", "bb", "s06", "pullback"}
        if strat in single_strats:
            ticker = args.ticker or "NVDA"
            capital = args.capital or 10000.0
            run_single_stock_pipeline(
                ticker=ticker,
                strategy_code=strat,
                lookback_years=years,
                eval_days=eval_days,
                initial_capital=capital,
                auto_chart=auto_chart,
            )
        else:
            capital = args.capital or 100000.0
            universe = NASDAQ_TOP_50_TICKERS if args.nasdaq_top50 else None
            run_portfolio_pipeline(
                strategy_code=strat,
                lookback_years=years,
                eval_days=eval_days,
                initial_capital=capital,
                universe=universe,
                auto_chart=auto_chart,
            )


if __name__ == "__main__":
    main()
