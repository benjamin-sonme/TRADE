"""
S04: Relative Momentum Rotation Strategy (相对动量轮动策略)
Periodically compares momentum across a candidate basket and allocates capital to the strongest assets.
- Monthly rebalance on the last trading day of the month.
- Momentum Formula (12-1 Momentum):
    M_t = (P_{t-21} / P_{t-252}) - 1
  (Uses total return price series, skips the most recent 21 trading days to avoid short-term mean reversion).
- Selection: Rank all candidates descending by M_t. Select top 4 stocks with M_t > 0.
- Allocation: 20% per selected stock, remaining capital (at least 20%) held in CASH.
- Execution: Rebalances on the first trading day of the next month.
"""

from typing import List, Dict
import pandas as pd
import numpy as np


class RelativeMomentumRotation:
    """
    Relative Momentum Rotation Strategy across a basket of stocks.
    """

    def __init__(
        self,
        lookback_days: int = 252,
        skip_recent_days: int = 21,
        top_n: int = 4,
        target_weight_per_stock: float = 0.20,
        min_cash_pct: float = 0.20,
    ):
        self.lookback_days = lookback_days
        self.skip_recent_days = skip_recent_days
        self.top_n = top_n
        self.target_weight_per_stock = target_weight_per_stock
        self.min_cash_pct = min_cash_pct
        self.name = f"RelativeMomentum_Top{top_n}_12_1M"

    def compute_target_weights(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes monthly target weights for each asset.
        Returns a DataFrame indexed by rebalance date with target weights per asset.
        """
        assets = list(prices_df.columns)
        
        # Identify month-end trading days
        # Group by year and month, take the last date of each month
        dates = pd.Series(prices_df.index, index=prices_df.index)
        month_end_dates = dates.groupby([dates.dt.year, dates.dt.month]).last().values

        records = []
        valid_rebalance_dates = []

        for date in month_end_dates:
            loc = prices_df.index.get_loc(date)
            # Need at least skip_recent_days to evaluate anything
            if loc < self.skip_recent_days + 20:
                continue

            momentum_scores = {}
            for asset in assets:
                series = prices_df[asset].dropna()
                if date not in series.index:
                    continue
                a_loc = series.index.get_loc(date)
                
                # Check if asset has at least lookback_days history
                if a_loc >= self.lookback_days:
                    p_21 = series.iloc[a_loc - self.skip_recent_days]
                    p_252 = series.iloc[a_loc - self.lookback_days]
                    if pd.notna(p_21) and pd.notna(p_252) and p_252 > 0:
                        momentum_scores[asset] = (p_21 / p_252) - 1.0
                elif a_loc >= self.skip_recent_days + 40:
                    # Adaptive momentum for newer IPOs with >40 days history
                    p_21 = series.iloc[a_loc - self.skip_recent_days]
                    p_start = series.iloc[0]
                    if pd.notna(p_21) and pd.notna(p_start) and p_start > 0:
                        momentum_scores[asset] = (p_21 / p_start) - 1.0

            if not momentum_scores:
                continue

            # Sort by score descending
            sorted_assets = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)

            # Filter top_n with positive momentum (M_t > 0)
            selected = [asset for asset, score in sorted_assets[: self.top_n] if score > 0]

            # Assign weights
            weight_dict = {a: 0.0 for a in assets}
            for a in selected:
                weight_dict[a] = self.target_weight_per_stock

            valid_rebalance_dates.append(date)
            records.append(weight_dict)

        weights_df = pd.DataFrame(records, index=valid_rebalance_dates)
        return weights_df
