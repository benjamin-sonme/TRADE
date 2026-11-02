"""
S06: Trend Pullback Strategy (趋势回撤买入策略)
Buys short-term pullbacks within confirmed long-term uptrends.
- Condition A (Uptrend): SMA(50) > SMA(200) and Close > SMA(200).
- Condition B (Pullback): In past 5 trading days, at least one day's Low <= SMA(50).
- Condition C (Reversal): Yesterday Close <= SMA(20) and Today Close > SMA(20).
- Buy Signal: A and B and C all satisfied simultaneously.
- Sell Signal: Consecutive 2 trading days Close < SMA(50).
"""

import pandas as pd
import numpy as np
from .base import BaseStrategy


class TrendPullbackStrategy(BaseStrategy):
    """
    Trend Pullback Strategy.
    Waits for a healthy dip to the 50-day moving average in an established uptrend,
    then enters when price reclaims the 20-day moving average.
    """

    def __init__(
        self,
        sma_fast: int = 20,
        sma_mid: int = 50,
        sma_slow: int = 200,
        pullback_window: int = 5,
    ):
        super().__init__(
            name=f"TrendPullback_{sma_fast}_{sma_mid}_{sma_slow}",
            sma_fast=sma_fast,
            sma_mid=sma_mid,
            sma_slow=sma_slow,
            pullback_window=pullback_window,
        )
        self.sma_fast = sma_fast
        self.sma_mid = sma_mid
        self.sma_slow = sma_slow
        self.pullback_window = pullback_window

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        # Moving Averages
        data["SMA_Fast"] = data["Close"].rolling(window=self.sma_fast).mean()
        data["SMA_Mid"] = data["Close"].rolling(window=self.sma_mid).mean()
        data["SMA_Slow"] = data["Close"].rolling(window=self.sma_slow).mean()

        # Condition A: Long-term uptrend
        cond_a = (data["SMA_Mid"] > data["SMA_Slow"]) & (data["Close"] > data["SMA_Slow"])

        # Condition B: Low touched or broke SMA_Mid in past `pullback_window` days
        touched_mid = data["Low"] <= data["SMA_Mid"]
        cond_b = touched_mid.rolling(window=self.pullback_window).max() == 1.0

        # Condition C: Regaining strength above SMA_Fast
        yesterday_below_fast = data["Close"].shift(1) <= data["SMA_Fast"].shift(1)
        today_above_fast = data["Close"] > data["SMA_Fast"]
        cond_c = yesterday_below_fast & today_above_fast

        # Exit Condition: 2 consecutive closes below SMA_Mid
        close_below_mid = data["Close"] < data["SMA_Mid"]
        exit_cond = close_below_mid & close_below_mid.shift(1)

        # Stateful Position Generation
        positions = np.zeros(len(data))
        current_pos = 0

        for i in range(len(data)):
            # Ensure indicators are calculated
            if np.isnan(data["SMA_Slow"].iloc[i]):
                positions[i] = 0
                continue

            a = cond_a.iloc[i]
            b = cond_b.iloc[i]
            c = cond_c.iloc[i]
            ex = exit_cond.iloc[i]

            if current_pos == 0:
                if a and b and c:
                    current_pos = 1
            elif current_pos == 1:
                if ex:
                    current_pos = 0

            positions[i] = current_pos

        data["Signal"] = positions
        return data
