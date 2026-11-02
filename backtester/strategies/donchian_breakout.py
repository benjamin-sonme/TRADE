"""
S03: Donchian Channel Breakout Strategy (唐奇安通道突破策略)
Classic trend-following strategy (Turtle Trading style):
- Buy when Close breaks above the highest high of the past N days (e.g. 20 days).
- Sell (exit to cash) when Close drops below the lowest low of the past M days (e.g. 10 days).
"""

import pandas as pd
import numpy as np
from .base import BaseStrategy


class DonchianBreakoutStrategy(BaseStrategy):
    """
    Donchian Channel Breakout Strategy.
    - Entry: Close > max(High of previous `entry_window` days).
    - Exit: Close < min(Low of previous `exit_window` days).
    Uses previous bars' High/Low (shifted by 1) to completely eliminate lookahead bias.
    """

    def __init__(
        self,
        entry_window: int = 20,
        exit_window: int = 10,
        long_only: bool = True,
    ):
        super().__init__(
            name=f"DonchianBreakout_{entry_window}_{exit_window}",
            entry_window=entry_window,
            exit_window=exit_window,
            long_only=long_only,
        )
        self.entry_window = entry_window
        self.exit_window = exit_window
        self.long_only = long_only

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        # Channel Upper (past entry_window days, shifted by 1 to exclude current day)
        data["Upper_Channel"] = data["High"].shift(1).rolling(window=self.entry_window).max()

        # Channel Lower (past exit_window days, shifted by 1 to exclude current day)
        data["Lower_Channel"] = data["Low"].shift(1).rolling(window=self.exit_window).min()

        # Channel Midpoint for reference/visualization
        data["Mid_Channel"] = (data["Upper_Channel"] + data["Lower_Channel"]) / 2.0

        positions = np.zeros(len(data))
        current_pos = 0

        for i in range(len(data)):
            upper = data["Upper_Channel"].iloc[i]
            lower = data["Lower_Channel"].iloc[i]
            close = data["Close"].iloc[i]

            if np.isnan(upper) or np.isnan(lower):
                positions[i] = 0
                continue

            if current_pos == 0:
                # Look for breakout to enter Long
                if close > upper:
                    current_pos = 1
            elif current_pos == 1:
                # Look for breakdown to exit to cash
                if close < lower:
                    current_pos = 0

            positions[i] = current_pos

        data["Signal"] = positions
        return data
