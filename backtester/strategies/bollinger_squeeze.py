"""
S05: Bollinger Band Squeeze Strategy (布林带挤压突破策略)
Captures directional breakouts following periods of low volatility consolidation.
- Middle = SMA(20), Upper = SMA(20) + 2*sigma, Lower = SMA(20) - 2*sigma
- BandWidth (BW) = (Upper - Lower) / Middle
- Low Volatility Squeeze: BW is in the lowest 10% of the past 126 trading days.
- Buy Signal: Squeeze occurred within the last 5 days, and Close breaks above Upper Band.
- Sell Signal: Close drops below the 20-day SMA (Middle Band).
"""

import pandas as pd
import numpy as np
from .base import BaseStrategy


class BollingerBandSqueezeStrategy(BaseStrategy):
    """
    Bollinger Band Squeeze Strategy.
    - Tracks volatility compression via BandWidth percentiles (past 126 days).
    - Triggers Long entry on upside breakout above Upper Band after squeeze.
    - Exits when price falls back below Middle Band (SMA 20).
    """

    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        squeeze_lookback: int = 126,
        squeeze_percentile: float = 10.0,
        squeeze_recent_window: int = 5,
    ):
        super().__init__(
            name=f"BBSqueeze_{period}_{std_dev}std_Q{int(squeeze_percentile)}",
            period=period,
            std_dev=std_dev,
            squeeze_lookback=squeeze_lookback,
            squeeze_percentile=squeeze_percentile,
            squeeze_recent_window=squeeze_recent_window,
        )
        self.period = period
        self.std_dev = std_dev
        self.squeeze_lookback = squeeze_lookback
        self.squeeze_quantile = squeeze_percentile / 100.0
        self.squeeze_recent_window = squeeze_recent_window

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        # 1. Bollinger Bands
        data["BB_Middle"] = data["Close"].rolling(window=self.period).mean()
        rolling_std = data["Close"].rolling(window=self.period).std()
        data["BB_Upper"] = data["BB_Middle"] + (self.std_dev * rolling_std)
        data["BB_Lower"] = data["BB_Middle"] - (self.std_dev * rolling_std)

        # 2. BandWidth
        data["BandWidth"] = (data["BB_Upper"] - data["BB_Lower"]) / data["BB_Middle"].replace(0, np.nan)

        # 3. Rolling Squeeze Threshold (lowest 10% quantile over past 126 days, no lookahead)
        data["BW_Threshold"] = data["BandWidth"].rolling(window=self.squeeze_lookback).quantile(self.squeeze_quantile)
        data["Is_Squeeze"] = data["BandWidth"] <= data["BW_Threshold"]

        # Check if squeeze occurred in the past `squeeze_recent_window` days
        data["Recent_Squeeze"] = data["Is_Squeeze"].rolling(window=self.squeeze_recent_window).max() == 1.0

        # 4. Stateful Position Generation
        positions = np.zeros(len(data))
        current_pos = 0

        for i in range(len(data)):
            close = data["Close"].iloc[i]
            upper = data["BB_Upper"].iloc[i]
            middle = data["BB_Middle"].iloc[i]
            recent_squeeze = data["Recent_Squeeze"].iloc[i]

            if np.isnan(upper) or np.isnan(middle):
                positions[i] = 0
                continue

            if current_pos == 0:
                # Buy condition: recent low-volatility squeeze + breakout above Upper band
                if recent_squeeze and (close > upper):
                    current_pos = 1
            elif current_pos == 1:
                # Sell condition: price falls back below 20 SMA
                if close < middle:
                    current_pos = 0

            positions[i] = current_pos

        data["Signal"] = positions
        return data
