"""
RSI Mean Reversion Strategy (RSI 均值回归策略)
Buys when oversold (RSI < oversold_threshold), exits when overbought (RSI > overbought_threshold).
Optionally uses a long-term trend filter (e.g. Close > 200 SMA) to avoid buying falling knives.
"""

import pandas as pd
import numpy as np
from .base import BaseStrategy


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates the Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's exponential smoothing
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


class RSIStrategy(BaseStrategy):
    """
    RSI Mean Reversion Strategy.
    - Buy Signal: RSI drops below `oversold` threshold (e.g. 30).
    - Sell Signal: RSI rises above `overbought` threshold (e.g. 70).
    - Trend Filter: When enabled, only enters if Close > `trend_ma` period SMA (default 200).
    """

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        use_trend_filter: bool = True,
        trend_ma_period: int = 200,
    ):
        name_str = f"RSI_{period}_OS{int(oversold)}_OB{int(overbought)}"
        if use_trend_filter:
            name_str += f"_TrendFilter{trend_ma_period}"

        super().__init__(
            name=name_str,
            period=period,
            oversold=oversold,
            overbought=overbought,
            use_trend_filter=use_trend_filter,
            trend_ma_period=trend_ma_period,
        )
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.use_trend_filter = use_trend_filter
        self.trend_ma_period = trend_ma_period

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        # Calculate RSI
        data["RSI"] = calculate_rsi(data["Close"], period=self.period)

        # Trend filter
        if self.use_trend_filter:
            data["Trend_MA"] = data["Close"].rolling(window=self.trend_ma_period).mean()
            trend_ok = (data["Close"] > data["Trend_MA"]) | data["Trend_MA"].isna()
        else:
            trend_ok = pd.Series(True, index=data.index)

        # Stateful signal generation
        positions = np.zeros(len(data))
        current_pos = 0

        for i in range(len(data)):
            rsi_val = data["RSI"].iloc[i]
            is_trend_ok = trend_ok.iloc[i]

            if current_pos == 0:
                # Look for buy signal
                if rsi_val <= self.oversold and is_trend_ok:
                    current_pos = 1
            elif current_pos == 1:
                # Look for sell signal
                if rsi_val >= self.overbought:
                    current_pos = 0

            positions[i] = current_pos

        data["Signal"] = positions
        return data
