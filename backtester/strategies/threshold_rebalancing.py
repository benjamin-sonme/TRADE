"""
S07: Threshold Rebalancing Strategy (阈值再平衡策略)
Rebalances portfolio positions when asset allocations drift beyond defined tolerance bands,
rather than trading on arbitrary calendar days or emotional price swings.
- For each stock i:
    Threshold_i = max(0.02, 0.25 * TargetWeight_i)
    Upper Threshold = TargetWeight_i + Threshold_i
    Lower Threshold = TargetWeight_i - Threshold_i
- Checked on the last trading day of each month.
- If weight > Upper Threshold: Trim back to TargetWeight_i (sell to cash).
- If weight < Lower Threshold: Add back to TargetWeight_i (buy with cash, respecting minimum cash constraint).
- If within tolerance band: Do nothing (zero friction).
- Cash constraints: Target cash e.g. 15%, Min cash 10%, no margin.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from ..metrics import calculate_metrics
from ..portfolio_engine import PortfolioBacktestResult, PortfolioRebalanceEvent


class ThresholdRebalancing:
    """
    Threshold Rebalancing strategy for an asset portfolio.
    """

    def __init__(
        self,
        target_weights: Dict[str, float],
        target_cash_pct: float = 0.15,
        min_cash_pct: float = 0.10,
        band_pct: float = 0.25,  # 25% relative drift
        min_absolute_threshold: float = 0.02,  # at least 2 percentage points
    ):
        self.target_weights = target_weights
        self.target_cash_pct = target_cash_pct
        self.min_cash_pct = min_cash_pct
        self.band_pct = band_pct
        self.min_absolute_threshold = min_absolute_threshold
        self.name = f"ThresholdRebalance_Band{int(band_pct*100)}%"

        # Compute threshold bands for each asset
        self.bands = {}
        for asset, tw in target_weights.items():
            thresh = max(min_absolute_threshold, band_pct * tw)
            self.bands[asset] = {
                "target": tw,
                "upper": tw + thresh,
                "lower": max(0.0, tw - thresh),
                "threshold": thresh,
            }

    def simulate(
        self,
        prices_df: pd.DataFrame,
        initial_capital: float = 100000.0,
        slippage_bps: float = 5.0,
        commission_per_trade: float = 0.0,
        benchmark_series: Optional[pd.Series] = None,
        benchmark_name: str = "Benchmark",
        eval_days: Optional[int] = None,
    ) -> PortfolioBacktestResult:
        """
        Runs monthly threshold evaluation and daily portfolio valuation.
        """
        if eval_days is not None and 0 < eval_days < len(prices_df):
            prices_df = prices_df.iloc[-eval_days:].copy()
            if benchmark_series is not None:
                benchmark_series = benchmark_series.loc[prices_df.index]

        assets = list(self.target_weights.keys())
        dates = pd.Series(prices_df.index, index=prices_df.index)
        month_end_dates = set(dates.groupby([dates.dt.year, dates.dt.month]).last().values)

        slippage_pct = slippage_bps / 10000.0
        cash = initial_capital
        holdings = {a: 0.0 for a in assets}

        # Initial allocation on day 1 (only for assets currently trading)
        day0_prices = prices_df.iloc[0]
        active_day0_assets = [a for a in assets if pd.notna(day0_prices[a]) and day0_prices[a] > 0]
        
        for a in active_day0_assets:
            tw = self.target_weights[a]
            alloc_val = initial_capital * tw
            exec_p = day0_prices[a] * (1.0 + slippage_pct)
            shares = alloc_val / exec_p if exec_p > 0 else 0.0
            holdings[a] = shares
            cash -= shares * exec_p

        daily_equity = []
        daily_weights = []
        rebalance_events = []

        pending_rebalance = False

        for i, date in enumerate(prices_df.index):
            current_prices = prices_df.iloc[i]
            valid_assets = [a for a in assets if pd.notna(current_prices[a]) and current_prices[a] > 0]

            # Execute pending rebalance from yesterday's trigger
            if pending_rebalance:
                total_val = cash + sum(holdings[a] * current_prices[a] for a in valid_assets)
                old_equity = total_val
                
                # Phase 1: Trim overweighted assets first (sell to raise cash)
                for a in valid_assets:
                    curr_val = holdings[a] * current_prices[a]
                    curr_w = curr_val / total_val if total_val > 0 else 0.0
                    target_w = self.target_weights[a]

                    if curr_w > self.bands[a]["upper"]:
                        excess_val = curr_val - (total_val * target_w)
                        exec_p = current_prices[a] * (1.0 - slippage_pct)
                        shares_to_sell = excess_val / exec_p
                        shares_to_sell = min(shares_to_sell, holdings[a])

                        holdings[a] -= shares_to_sell
                        cash += (shares_to_sell * exec_p) - commission_per_trade

                # Phase 2: Add underweighted assets (if cash is above min_cash_pct)
                total_val = cash + sum(holdings[a] * current_prices[a] for a in valid_assets)
                for a in valid_assets:
                    curr_val = holdings[a] * current_prices[a]
                    curr_w = curr_val / total_val if total_val > 0 else 0.0
                    target_w = self.target_weights[a]

                    if curr_w < self.bands[a]["lower"]:
                        needed_val = (total_val * target_w) - curr_val
                        max_cash_allowed = max(0.0, cash - (total_val * self.min_cash_pct))
                        val_to_buy = min(needed_val, max_cash_allowed)

                        if val_to_buy > 0:
                            exec_p = current_prices[a] * (1.0 + slippage_pct)
                            shares_to_buy = (val_to_buy - commission_per_trade) / exec_p
                            if shares_to_buy > 0:
                                holdings[a] += shares_to_buy
                                cash -= (shares_to_buy * exec_p) + commission_per_trade

                new_equity = cash + sum(holdings[a] * current_prices[a] for a in valid_assets)
                rebalance_events.append(
                    PortfolioRebalanceEvent(
                        date=date,
                        target_weights=self.target_weights,
                        actual_weights={a: (holdings[a] * current_prices[a]) / new_equity for a in valid_assets},
                        portfolio_equity=new_equity,
                        cash=cash,
                        turnover=abs(new_equity - old_equity) / new_equity if new_equity > 0 else 0.0,
                    )
                )
                pending_rebalance = False

            # Check if today is month end -> inspect drift
            if date in month_end_dates:
                total_val = cash + sum(holdings[a] * current_prices[a] for a in valid_assets)
                needs_adjustment = False
                for a in valid_assets:
                    curr_w = (holdings[a] * current_prices[a]) / total_val if total_val > 0 else 0.0
                    if curr_w > self.bands[a]["upper"] or curr_w < self.bands[a]["lower"]:
                        needs_adjustment = True
                        break
                if needs_adjustment:
                    pending_rebalance = True

            day_equity = cash + sum(holdings[a] * current_prices[a] for a in valid_assets)
            daily_equity.append(day_equity)

            day_w = {a: (holdings[a] * current_prices[a]) / day_equity if day_equity > 0 else 0.0 for a in valid_assets}
            day_w["CASH"] = cash / day_equity if day_equity > 0 else 1.0
            daily_weights.append(day_w)

        equity_series = pd.Series(daily_equity, index=prices_df.index)
        weights_history = pd.DataFrame(daily_weights, index=prices_df.index)

        # Benchmark equity
        if benchmark_series is not None and not benchmark_series.empty:
            bench_aligned = benchmark_series.reindex(prices_df.index).ffill().bfill()
            bench_shares = initial_capital / bench_aligned.iloc[0]
            benchmark_equity = bench_shares * bench_aligned
        else:
            # Default benchmark: Buy & Hold equal weight of the assets
            norm_prices = prices_df.apply(lambda col: (col / col.dropna().iloc[0]) if len(col.dropna()) > 0 else 1.0)
            benchmark_equity = (initial_capital / len(assets)) * norm_prices.fillna(1.0).sum(axis=1)

        metrics = calculate_metrics(
            equity_series=equity_series,
            benchmark_equity=benchmark_equity,
        )

        equity_df = pd.DataFrame(
            {
                "Strategy_Equity": equity_series,
                "Benchmark_Equity": benchmark_equity,
            },
            index=prices_df.index,
        )

        return PortfolioBacktestResult(
            strategy_name=self.name,
            universe=assets,
            start_date=prices_df.index[0].strftime("%Y-%m-%d"),
            end_date=prices_df.index[-1].strftime("%Y-%m-%d"),
            initial_capital=initial_capital,
            final_equity=equity_series.iloc[-1],
            benchmark_name=benchmark_name,
            benchmark_final_equity=benchmark_equity.iloc[-1],
            metrics=metrics,
            equity_df=equity_df,
            weights_history=weights_history,
            rebalance_events=rebalance_events,
        )
