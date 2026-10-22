"""
Base Strategy Interface
Every strategy inherits from this class and implements `generate_signals`.
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, name: str, **params):
        self.name = name
        self.params = params

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes OHLCV dataframe and returns dataframe with indicator columns
        and a 'Signal' column where:
           1: Hold Long position
           0: Cash (Flat / Out of Market)
          -1: Hold Short position (if shorting is enabled)
        """
        pass

    def __repr__(self) -> str:
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({param_str})"
