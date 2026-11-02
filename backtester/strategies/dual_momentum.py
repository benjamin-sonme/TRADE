"""
S09: Dual Momentum Strategy (双重动量策略 - Gary Antonacci)
Combines Relative Momentum (which risk asset is stronger?)
with Absolute Momentum (is the risk asset outperforming cash/treasuries?).
- Core Assets:
    SPY: U.S. Large Cap Equities
    VEU: All-World ex-U.S. Equities
    BIL: 1-3 Month U.S. Treasury Bills (Cash / Safe Asset)
- Frequency: Monthly rebalance on the last trading day.
- Decision Logic (from Gary Antonacci / User Flowchart):
    1. Compare 12-month total return: R_{12m}(SPY) vs R_{12m}(VEU).
       Winner = argmax(R_{12m}(SPY), R_{12m}(VEU)).
    2. Check if Winner outpeforms BIL:
       If R_{12m}(Winner) > R_{12m}(BIL):
           Hold 100% Winner
       Else:
           Hold 100% BIL (Defensive Cash Mode)
- Execution: Rebalance executed on the 1st trading day of the next month.
"""

from typing import Dict, List
import pandas as pd
import numpy as np


class DualMomentumStrategy:
    """
    Dual Momentum Strategy (Gary Antonacci model).
    """

    def __init__(
        self,
        us_equity_ticker: str = "SPY",
        intl_equity_ticker: str = "VEU",
        safe_asset_ticker: str = "BIL",
        lookback_days: int = 252,  # 12 months (approx 252 trading days)
    ):
        self.us_equity = us_equity_ticker
        self.intl_equity = intl_equity_ticker
        self.safe_asset = safe_asset_ticker
        self.lookback_days = lookback_days
        self.assets = [self.us_equity, self.intl_equity, self.safe_asset]
        self.name = f"DualMomentum_{self.us_equity}_{self.intl_equity}_{self.safe_asset}"

    def compute_target_weights(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes monthly target weights (100% in one of SPY, VEU, or BIL).
        """
        dates = pd.Series(prices_df.index, index=prices_df.index)
        month_end_dates = dates.groupby([dates.dt.year, dates.dt.month]).last().values

        records = []
        valid_rebalance_dates = []

        for date in month_end_dates:
            loc = prices_df.index.get_loc(date)
            if loc < self.lookback_days:
                continue

            # Calculate 12-month returns
            r_us = (prices_df[self.us_equity].iloc[loc] / prices_df[self.us_equity].iloc[loc - self.lookback_days]) - 1.0
            r_intl = (prices_df[self.intl_equity].iloc[loc] / prices_df[self.intl_equity].iloc[loc - self.lookback_days]) - 1.0
            r_safe = (prices_df[self.safe_asset].iloc[loc] / prices_df[self.safe_asset].iloc[loc - self.lookback_days]) - 1.0

            # Step 1: Relative Momentum (SPY vs VEU)
            if r_us >= r_intl:
                winner_stock = self.us_equity
                winner_return = r_us
            else:
                winner_stock = self.intl_equity
                winner_return = r_intl

            # Step 2: Absolute Momentum (Winner vs BIL)
            weight_dict = {self.us_equity: 0.0, self.intl_equity: 0.0, self.safe_asset: 0.0}
            if winner_return > r_safe:
                # Hold the stronger risk asset
                weight_dict[winner_stock] = 1.0
            else:
                # Shift to defensive safe asset
                weight_dict[self.safe_asset] = 1.0

            valid_rebalance_dates.append(date)
            records.append(weight_dict)

        weights_df = pd.DataFrame(records, index=valid_rebalance_dates)
        return weights_df
