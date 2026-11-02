"""
Portfolio Backtesting Engine Module
Simulates multi-asset portfolio strategies with dynamic target weights,
cash management, rebalancing friction, and benchmark comparisons.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from .metrics import calculate_metrics


@dataclass
class PortfolioRebalanceEvent:
    date: pd.Timestamp
    target_weights: Dict[str, float]
    actual_weights: Dict[str, float]
    portfolio_equity: float
    cash: float
    turnover: float
    notes: str = ""


@dataclass
class PortfolioBacktestResult:
    strategy_name: str
    universe: List[str]
    start_date: str
    end_date: str
    initial_capital: float
    final_equity: float
    benchmark_name: str
    benchmark_final_equity: float
    metrics: Dict[str, Any]
    equity_df: pd.DataFrame
    weights_history: pd.DataFrame
    rebalance_events: List[PortfolioRebalanceEvent]


class PortfolioEngine:
    """
    Engine for simulating multi-asset portfolio strategies.
    Takes a dictionary or multi-column DataFrame of historical prices (Close)
    and executes weight-based rebalancing without lookahead bias.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_per_trade: float = 0.0,
        slippage_bps: float = 5.0,  # 5 bps
        risk_free_rate: float = 0.04,
    ):
        self.initial_capital = initial_capital
        self.commission_per_trade = commission_per_trade
        self.slippage_pct = slippage_bps / 10000.0
        self.risk_free_rate = risk_free_rate

    def run(
        self,
        prices_df: pd.DataFrame,
        target_weights_df: pd.DataFrame,
        strategy_name: str = "PortfolioStrategy",
        benchmark_series: Optional[pd.Series] = None,
        benchmark_name: str = "Benchmark",
    ) -> PortfolioBacktestResult:
        """
        Runs portfolio simulation.
        - prices_df: DataFrame of daily Close prices where columns are asset tickers.
        - target_weights_df: DataFrame of target weights on each rebalance date.
          Index contains rebalance dates (e.g. month ends).
          Columns are asset tickers (weights should sum to <= 1.0; remainder is cash).
        """
        # Ensure dates align
        common_index = prices_df.index
        assets = list(prices_df.columns)

        # Execution happens on the day AFTER the signal/target weight date (zero lookahead)
        # Shift target weights: decision at date T takes effect at date T+1
        rebalance_dates = set(target_weights_df.index)

        cash = self.initial_capital
        holdings = {asset: 0.0 for asset in assets}  # number of shares

        daily_equity = []
        daily_weights = []
        rebalance_events: List[PortfolioRebalanceEvent] = []

        pending_target_weights: Optional[Dict[str, float]] = None

        for i, date in enumerate(common_index):
            current_prices = prices_df.iloc[i]

            # 1. Execute any pending rebalance from yesterday's decision
            if pending_target_weights is not None:
                current_total_value = cash + sum(holdings[a] * current_prices[a] for a in assets if current_prices[a] > 0)
                old_holdings_val = sum(holdings[a] * current_prices[a] for a in assets if current_prices[a] > 0)

                # Sell first to raise cash
                for a in assets:
                    t_weight = pending_target_weights.get(a, 0.0)
                    target_val = current_total_value * t_weight
                    curr_val = holdings[a] * current_prices[a]

                    if curr_val > target_val and current_prices[a] > 0:
                        val_to_sell = curr_val - target_val
                        exec_price = current_prices[a] * (1.0 - self.slippage_pct)
                        shares_to_sell = val_to_sell / exec_price
                        shares_to_sell = min(shares_to_sell, holdings[a])

                        holdings[a] -= shares_to_sell
                        cash += (shares_to_sell * exec_price) - self.commission_per_trade

                # Buy next using available cash
                for a in assets:
                    t_weight = pending_target_weights.get(a, 0.0)
                    target_val = current_total_value * t_weight
                    curr_val = holdings[a] * current_prices[a]

                    if curr_val < target_val and current_prices[a] > 0:
                        val_to_buy = min(target_val - curr_val, max(0.0, cash))
                        if val_to_buy > 0:
                            exec_price = current_prices[a] * (1.0 + self.slippage_pct)
                            shares_to_buy = (val_to_buy - self.commission_per_trade) / exec_price
                            if shares_to_buy > 0:
                                holdings[a] += shares_to_buy
                                cash -= (shares_to_buy * exec_price) + self.commission_per_trade

                new_equity = cash + sum(holdings[a] * current_prices[a] for a in assets)
                actual_w = {a: (holdings[a] * current_prices[a]) / new_equity for a in assets}
                actual_w["CASH"] = cash / new_equity

                rebalance_events.append(
                    PortfolioRebalanceEvent(
                        date=date,
                        target_weights=pending_target_weights,
                        actual_weights=actual_w,
                        portfolio_equity=new_equity,
                        cash=cash,
                        turnover=abs(new_equity - old_holdings_val) / new_equity,
                    )
                )
                pending_target_weights = None

            # 2. Check if today generates a new target weight for tomorrow
            if date in rebalance_dates:
                w_row = target_weights_df.loc[date]
                pending_target_weights = {a: float(w_row.get(a, 0.0)) for a in assets}

            # 3. Calculate portfolio equity at close of today
            day_equity = cash + sum(holdings[a] * current_prices[a] for a in assets if not np.isnan(current_prices[a]))
            daily_equity.append(day_equity)

            # Record weight distribution for today
            day_w = {a: (holdings[a] * current_prices[a]) / day_equity if day_equity > 0 else 0.0 for a in assets}
            day_w["CASH"] = cash / day_equity if day_equity > 0 else 1.0
            daily_weights.append(day_w)

        equity_series = pd.Series(daily_equity, index=common_index)
        weights_history = pd.DataFrame(daily_weights, index=common_index)

        # Benchmark equity
        if benchmark_series is not None and not benchmark_series.empty:
            bench_aligned = benchmark_series.reindex(common_index).ffill()
            bench_shares = self.initial_capital / bench_aligned.iloc[0]
            benchmark_equity = bench_shares * bench_aligned
        else:
            # Default benchmark: equal-weight buy & hold of the universe
            norm_prices = prices_df.divide(prices_df.iloc[0])
            benchmark_equity = (self.initial_capital / len(assets)) * norm_prices.sum(axis=1)

        # Performance metrics
        metrics = calculate_metrics(
            equity_series=equity_series,
            benchmark_equity=benchmark_equity,
            risk_free_rate=self.risk_free_rate,
        )

        equity_df = pd.DataFrame(
            {
                "Strategy_Equity": equity_series,
                "Benchmark_Equity": benchmark_equity,
            },
            index=common_index,
        )

        return PortfolioBacktestResult(
            strategy_name=strategy_name,
            universe=assets,
            start_date=common_index[0].strftime("%Y-%m-%d"),
            end_date=common_index[-1].strftime("%Y-%m-%d"),
            initial_capital=self.initial_capital,
            final_equity=equity_series.iloc[-1],
            benchmark_name=benchmark_name,
            benchmark_final_equity=benchmark_equity.iloc[-1],
            metrics=metrics,
            equity_df=equity_df,
            weights_history=weights_history,
            rebalance_events=rebalance_events,
        )
