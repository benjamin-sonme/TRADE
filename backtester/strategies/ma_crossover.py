"""
Dual Moving Average Crossover Strategy (双均线交叉策略)
Standard trend-following strategy:
- Buy when Short MA crosses above Long MA (Golden Cross / 金叉).
- Sell to cash when Short MA crosses below Long MA (Death Cross / 死叉).
"""

import pandas as pd
import numpy as np
from .base import BaseStrategy


class DualMACrossoverStrategy(BaseStrategy):
    """
    Dual Moving Average Crossover Strategy.
    Supports Simple Moving Average (SMA) and Exponential Moving Average (EMA).
    """

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 50,
        ma_type: str = "SMA",
        long_only: bool = True,
    ):
        super().__init__(
            name=f"DualMA_{ma_type}_{short_window}_{long_window}",
            short_window=short_window,
            long_window=long_window,
            ma_type=ma_type.upper(),
            long_only=long_only,
        )
        self.short_window = short_window
        self.long_window = long_window
        self.ma_type = ma_type.upper()
        self.long_only = long_only

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        # Calculate moving averages
        if self.ma_type == "EMA":
            data["MA_Short"] = data["Close"].ewm(span=self.short_window, adjust=False).mean()
            data["MA_Long"] = data["Close"].ewm(span=self.long_window, adjust=False).mean()
        else:
            data["MA_Short"] = data["Close"].rolling(window=self.short_window).mean()
            data["MA_Long"] = data["Close"].rolling(window=self.long_window).mean()

        # Generate raw position signal
        # 1 when Short > Long, 0 or -1 otherwise
        signals = np.zeros(len(data))
        valid_mask = data["MA_Short"].notna() & data["MA_Long"].notna()

        if self.long_only:
            signals = np.where(valid_mask & (data["MA_Short"] > data["MA_Long"]), 1, 0)
        else:
            signals = np.where(
                valid_mask & (data["MA_Short"] > data["MA_Long"]),
                1,
                np.where(valid_mask & (data["MA_Short"] < data["MA_Long"]), -1, 0),
            )

        data["Signal"] = signals
        return data
