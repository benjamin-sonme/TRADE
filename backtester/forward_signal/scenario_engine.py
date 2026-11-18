"""
Forward Scenario Engine (Single Source of Truth).

Constructs hypothetical next-trading-day market scenarios and invokes the EXACT
same strategy classes (`generate_signals` / `compute_target_weights` / `simulate`)
used by the historical backtest engine, guaranteeing 100% behavioral parity.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from ..strategies.ma_crossover import DualMACrossoverStrategy
from ..strategies.rsi_strategy import RSIStrategy
from ..strategies.donchian_breakout import DonchianBreakoutStrategy
from ..strategies.bollinger_squeeze import BollingerBandSqueezeStrategy
from ..strategies.trend_pullback import TrendPullbackStrategy
from ..strategies.relative_momentum import RelativeMomentumRotation
from ..strategies.threshold_rebalancing import ThresholdRebalancing
from ..strategies.quality_value_factor import QualityValueFactorStrategy, DEFAULT_FACTOR_UNIVERSE
from ..strategies.dual_momentum import DualMomentumStrategy


STRATEGY_METADATA: Dict[str, Dict[str, Any]] = {
    "S01": {
        "cn_name": "双均线交叉策略",
        "en_name": "Dual MA Crossover",
        "is_portfolio": False,
        "category": "EXACT_THRESHOLD",
        "category_label": "数学反推精确阈值",
    },
    "S02": {
        "cn_name": "RSI 均值回归策略",
        "en_name": "RSI Mean Reversion",
        "is_portfolio": False,
        "category": "EXACT_THRESHOLD",
        "category_label": "递推公式反推阈值",
    },
    "S03": {
        "cn_name": "唐奇安通道突破策略",
        "en_name": "Donchian Breakout",
        "is_portfolio": False,
        "category": "EXACT_THRESHOLD",
        "category_label": "历史已知阈值",
    },
    "S04": {
        "cn_name": "相对动量轮动策略",
        "en_name": "Relative Momentum Rotation",
        "is_portfolio": True,
        "category": "DYNAMIC_SCENARIO",
        "category_label": "多资产排名与调仓日情景",
    },
    "S05": {
        "cn_name": "布林带挤压突破策略",
        "en_name": "Bollinger Band Squeeze",
        "is_portfolio": False,
        "category": "DYNAMIC_SCENARIO",
        "category_label": "动态指标模型模拟",
    },
    "S06": {
        "cn_name": "趋势回撤买入策略",
        "en_name": "Trend Pullback",
        "is_portfolio": False,
        "category": "EXACT_THRESHOLD",
        "category_label": "均线与状态条件反推",
    },
    "S07": {
        "cn_name": "阈值动态再平衡策略",
        "en_name": "Threshold Rebalancing",
        "is_portfolio": True,
        "category": "DYNAMIC_SCENARIO",
        "category_label": "组合权重容忍带与调仓日",
    },
    "S08": {
        "cn_name": "质量与价值因子策略",
        "en_name": "Quality & Value Factor",
        "is_portfolio": True,
        "category": "FUNDAMENTAL_BOUNDED",
        "category_label": "基本面财报主导（非纯价格触发）",
    },
    "S09": {
        "cn_name": "双重动量资产轮动策略",
        "en_name": "Dual Momentum",
        "is_portfolio": True,
        "category": "DYNAMIC_SCENARIO",
        "category_label": "多ETF相对/绝对动量情景",
    },
}


def create_single_stock_strategy(strategy_code: str):
    """Canonical strategy factory shared by Historical Backtest and Next-Day Signal Preview."""
    code = strategy_code.upper()
    if code == "S01":
        return DualMACrossoverStrategy(short_window=20, long_window=50)
    if code == "S02":
        return RSIStrategy(
            period=14,
            oversold=30.0,
            overbought=70.0,
            use_trend_filter=True,
            trend_ma_period=200,
        )
    if code == "S03":
        return DonchianBreakoutStrategy(entry_window=20, exit_window=10)
    if code == "S05":
        return BollingerBandSqueezeStrategy(
            period=20,
            std_dev=2.0,
            squeeze_lookback=126,
            squeeze_percentile=10.0,
            squeeze_recent_window=5,
        )
    if code == "S06":
        return TrendPullbackStrategy(
            sma_fast=20,
            sma_mid=50,
            sma_slow=200,
            pullback_window=5,
        )
    raise ValueError(f"Unsupported single-stock strategy code: {strategy_code}")


def create_portfolio_strategy(strategy_code: str, universe: List[str]):
    """Canonical portfolio strategy factory shared by Historical Backtest and Next-Day Signal Preview."""
    code = strategy_code.upper()
    if code == "S04":
        return RelativeMomentumRotation(top_n=4, target_weight_per_stock=0.20)
    if code == "S07":
        clean_u = list(universe) if universe else ["NVDA", "AAPL", "MSFT", "AMZN"]
        tw_per_stock = 0.80 / len(clean_u)
        target_weights = {asset: tw_per_stock for asset in clean_u}
        return ThresholdRebalancing(target_weights=target_weights, target_cash_pct=0.20, band_pct=0.25)
    if code == "S08":
        clean_u = list(universe) if universe else DEFAULT_FACTOR_UNIVERSE
        return QualityValueFactorStrategy(universe=clean_u, top_n=5, target_weight_per_stock=0.15)
    if code == "S09":
        return DualMomentumStrategy()
    raise ValueError(f"Unsupported portfolio strategy code: {strategy_code}")


def estimate_next_trading_date(as_of_ts: pd.Timestamp) -> pd.Timestamp:
    """Returns the next weekday trading date (skips Saturday/Sunday)."""
    ts = pd.Timestamp(as_of_ts)
    return ts + BDay(1)


def is_month_end_business_day(ts: pd.Timestamp) -> bool:
    """True if `ts` is the last weekday (Mon-Fri) of its calendar month."""
    ts = pd.Timestamp(ts)
    next_bday = ts + BDay(1)
    return next_bday.month != ts.month


def next_month_end_business_day(as_of_ts: pd.Timestamp, include_today: bool = False) -> pd.Timestamp:
    """Returns the next month-end weekday trading date on or after `as_of_ts`."""
    cur = pd.Timestamp(as_of_ts)
    if not include_today:
        cur = cur + BDay(1)
    while not is_month_end_business_day(cur):
        cur = cur + BDay(1)
    return cur


def classify_transition(prev_pos: int, new_pos: int) -> str:
    """Maps a position state transition to BUY / SELL / HOLD / WAIT."""
    if prev_pos == 0 and new_pos == 1:
        return "BUY"
    if prev_pos == 1 and new_pos == 0:
        return "SELL"
    if new_pos == 1:
        return "HOLD"
    return "WAIT"


class ForwardScenarioEngine:
    """
    Appends synthetic next-day market bar(s) to historical OHLCV data and executes
    the real strategy engine to evaluate what signal the strategy would produce.
    """

    @staticmethod
    def evaluate_single_stock_history(df: pd.DataFrame, strategy) -> pd.DataFrame:
        """Runs the exact strategy signal generator on completed historical data."""
        if df is None or df.empty:
            raise ValueError("Historical OHLCV DataFrame is empty.")
        return strategy.generate_signals(df)

    @staticmethod
    def append_next_day_bar(
        df: pd.DataFrame,
        close: float,
        open_px: Optional[float] = None,
        high: Optional[float] = None,
        low: Optional[float] = None,
        volume: Optional[float] = None,
        next_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Appends a single hypothetical next-trading-day OHLCV row to `df`."""
        last_row = df.iloc[-1]
        last_close = float(last_row["Close"])
        target_dt = pd.Timestamp(next_date) if next_date is not None else estimate_next_trading_date(df.index[-1])

        o = float(open_px) if open_px is not None else float(close)
        c = float(close)
        h = float(high) if high is not None else max(o, c)
        l = float(low) if low is not None else min(o, c)
        v = float(volume) if volume is not None else float(last_row.get("Volume", 0.0))

        row_dict: Dict[str, Any] = {}
        for col in df.columns:
            if col == "Open":
                row_dict[col] = o
            elif col == "High":
                row_dict[col] = h
            elif col == "Low":
                row_dict[col] = l
            elif col == "Close":
                row_dict[col] = c
            elif col == "Volume":
                row_dict[col] = v
            else:
                row_dict[col] = last_row[col]

        # Build compact new frame without mutating caller's DataFrame
        new_row = pd.DataFrame([row_dict], index=pd.DatetimeIndex([target_dt], name=df.index.name))
        return pd.concat([df, new_row])

    @classmethod
    def evaluate_next_day_price(
        cls,
        df: pd.DataFrame,
        strategy,
        test_close: float,
        test_low: Optional[float] = None,
        test_high: Optional[float] = None,
        test_open: Optional[float] = None,
        next_date: Optional[pd.Timestamp] = None,
    ) -> Tuple[int, int, str, Dict[str, Any]]:
        """
        Simulates appending a hypothetical next-day bar to `df` and invokes
        `strategy.generate_signals(sim_df)` directly.

        Returns:
            (new_pos, prev_pos, action_str, next_day_row_dict)
        """
        sim_df = cls.append_next_day_bar(
            df=df,
            close=test_close,
            open_px=test_open,
            high=test_high,
            low=test_low,
            next_date=next_date,
        )
        sig_df = strategy.generate_signals(sim_df)
        prev_pos = int(sig_df["Signal"].iloc[-2]) if len(sig_df) >= 2 else 0
        new_pos = int(sig_df["Signal"].iloc[-1])
        action = classify_transition(prev_pos, new_pos)

        last = sig_df.iloc[-1]
        snapshot: Dict[str, Any] = {}
        for col in sig_df.columns:
            val = last[col]
            if isinstance(val, (np.floating, float)):
                snapshot[col] = None if np.isnan(val) else float(val)
            elif isinstance(val, (np.integer, int)):
                snapshot[col] = int(val)
            elif isinstance(val, (np.bool_, bool)):
                snapshot[col] = bool(val)
            else:
                snapshot[col] = val

        return new_pos, prev_pos, action, snapshot

    @staticmethod
    def append_next_day_multi_prices(
        prices_df: pd.DataFrame,
        override_prices: Optional[Dict[str, float]] = None,
        next_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Appends one hypothetical next trading day to a multi-asset close price DataFrame."""
        target_dt = pd.Timestamp(next_date) if next_date is not None else estimate_next_trading_date(prices_df.index[-1])
        last_prices = prices_df.iloc[-1].copy()
        if override_prices:
            for sym, px in override_prices.items():
                if sym in last_prices.index:
                    last_prices[sym] = float(px)
        new_row = pd.DataFrame([last_prices.to_dict()], index=pd.DatetimeIndex([target_dt], name=prices_df.index.name))
        return pd.concat([prices_df, new_row])
