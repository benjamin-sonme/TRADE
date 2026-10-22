"""
Performance Metrics Calculation Module
Computes cumulative return, Sharpe ratio, max drawdown, CAGR, win rate, and benchmark comparisons.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd


def calculate_metrics(
    equity_series: pd.Series,
    benchmark_equity: pd.Series,
    trade_returns: List[float] = None,
    risk_free_rate: float = 0.04,
    trading_days_per_year: int = 252,
) -> Dict[str, Any]:
    """
    Computes key performance metrics for strategy equity vs benchmark equity.
    """
    if len(equity_series) < 2:
        return {}

    # Daily returns
    strat_daily_returns = equity_series.pct_change().dropna()
    bench_daily_returns = benchmark_equity.pct_change().dropna()

    total_days = len(strat_daily_returns)
    years = total_days / trading_days_per_year

    # 1. Cumulative Return
    strat_total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1.0
    bench_total_return = (benchmark_equity.iloc[-1] / benchmark_equity.iloc[0]) - 1.0

    # 2. Annualized Return (CAGR)
    if years > 0 and (equity_series.iloc[-1] / equity_series.iloc[0]) > 0:
        strat_cagr = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (1.0 / years) - 1.0
    else:
        strat_cagr = np.nan

    if years > 0 and (benchmark_equity.iloc[-1] / benchmark_equity.iloc[0]) > 0:
        bench_cagr = (benchmark_equity.iloc[-1] / benchmark_equity.iloc[0]) ** (1.0 / years) - 1.0
    else:
        bench_cagr = np.nan

    # 3. Annualized Volatility
    strat_vol = strat_daily_returns.std() * np.sqrt(trading_days_per_year)
    bench_vol = bench_daily_returns.std() * np.sqrt(trading_days_per_year)

    # 4. Sharpe Ratio
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / trading_days_per_year) - 1.0
    strat_excess = strat_daily_returns - daily_rf
    bench_excess = bench_daily_returns - daily_rf

    strat_sharpe = (
        (strat_excess.mean() / strat_daily_returns.std() * np.sqrt(trading_days_per_year))
        if strat_daily_returns.std() > 0
        else 0.0
    )
    bench_sharpe = (
        (bench_excess.mean() / bench_daily_returns.std() * np.sqrt(trading_days_per_year))
        if bench_daily_returns.std() > 0
        else 0.0
    )

    # 5. Drawdowns
    strat_cummax = equity_series.cummax()
    strat_dd = (equity_series - strat_cummax) / strat_cummax
    strat_max_dd = strat_dd.min()

    bench_cummax = benchmark_equity.cummax()
    bench_dd = (benchmark_equity - bench_cummax) / bench_cummax
    bench_max_dd = bench_dd.min()

    # Max Drawdown Duration (days)
    is_in_dd = strat_dd < 0
    dd_durations = []
    current_dd_len = 0
    for in_dd in is_in_dd:
        if in_dd:
            current_dd_len += 1
        else:
            if current_dd_len > 0:
                dd_durations.append(current_dd_len)
            current_dd_len = 0
    if current_dd_len > 0:
        dd_durations.append(current_dd_len)
    max_dd_duration = max(dd_durations) if dd_durations else 0

    # 6. Trade-level statistics
    total_trades = len(trade_returns) if trade_returns is not None else 0
    if total_trades > 0:
        wins = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r <= 0]
        win_rate = len(wins) / total_trades
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (np.inf if gross_profit > 0 else 0.0)
    else:
        win_rate = 0.0
        profit_factor = 0.0

    return {
        "Strategy Total Return": strat_total_return,
        "Benchmark Total Return": bench_total_return,
        "Strategy CAGR": strat_cagr,
        "Benchmark CAGR": bench_cagr,
        "Strategy Volatility": strat_vol,
        "Benchmark Volatility": bench_vol,
        "Strategy Sharpe": strat_sharpe,
        "Benchmark Sharpe": bench_sharpe,
        "Strategy Max Drawdown": strat_max_dd,
        "Benchmark Max Drawdown": bench_max_dd,
        "Max DD Duration (Days)": max_dd_duration,
        "Total Trades": total_trades,
        "Win Rate": win_rate,
        "Profit Factor": profit_factor,
        "Drawdown Series": strat_dd,
        "Benchmark Drawdown Series": bench_dd,
    }
