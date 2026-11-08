"""
Backtesting Engine Module
Simulates strategy execution, calculates realistic portfolio equity with slippage/commissions,
and computes comparative performance against Buy & Hold.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from .strategies.base import BaseStrategy
from .metrics import calculate_metrics


@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    position_type: str  # "LONG" or "SHORT"
    pnl: float
    return_pct: float
    holding_days: int


@dataclass
class BacktestResult:
    ticker: str
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    final_equity: float
    benchmark_final_equity: float
    metrics: Dict[str, Any]
    equity_df: pd.DataFrame
    trades: List[Trade]


class BacktestEngine:
    """
    Simulates trading strategies on historical data with realistic assumptions:
    - Zero lookahead bias: Signals generated at Close(T) execute on Open/Close(T+1).
    - Customizable commission and slippage (in basis points).
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_per_trade: float = 0.0,
        slippage_bps: float = 5.0,  # 5 bps = 0.05%
        risk_free_rate: float = 0.04,
    ):
        self.initial_capital = initial_capital
        self.commission_per_trade = commission_per_trade
        self.slippage_pct = slippage_bps / 10000.0
        self.risk_free_rate = risk_free_rate

    def run(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        ticker: str = "TICKER",
        eval_days: Optional[int] = None,
    ) -> BacktestResult:
        """Runs the backtest on the given DataFrame using the given Strategy."""
        # 1. Generate signals on full warmed-up history
        data = strategy.generate_signals(df)

        if "Signal" not in data.columns:
            raise ValueError(f"Strategy {strategy.name} must produce a 'Signal' column.")

        # 2. Shift signals by 1 to prevent lookahead bias (Execution happens after signal bar)
        data["Position"] = data["Signal"].shift(1).fillna(0)

        # 3. Slice to evaluation window if requested (e.g. last 3 days, 1 week, 1 month, 6 months)
        if eval_days is not None and eval_days < len(data) and eval_days > 0:
            data = data.iloc[-eval_days:].copy()

        # Detect position changes (trades) in the evaluation window
        data["Position_Change"] = data["Position"].diff().fillna(data["Position"])

        # 3. Simulate day-by-day cash, holdings, and portfolio equity
        cash = self.initial_capital
        shares = 0.0
        portfolio_equity = []
        trades: List[Trade] = []

        current_entry_date = None
        current_entry_price = 0.0
        current_entry_shares = 0.0

        for i in range(len(data)):
            row = data.iloc[i]
            date = data.index[i]
            price = row["Close"]
            target_pos = row["Position"]
            pos_change = row["Position_Change"]

            # If position changes, execute trade
            if pos_change != 0:
                if target_pos == 1:
                    # Enter Long
                    exec_price = price * (1.0 + self.slippage_pct)
                    available_cash = max(0.0, cash - self.commission_per_trade)
                    shares_to_buy = available_cash / exec_price if exec_price > 0 else 0.0

                    shares = shares_to_buy
                    cash = available_cash - (shares_to_buy * exec_price)
                    current_entry_date = date
                    current_entry_price = exec_price
                    current_entry_shares = shares

                elif target_pos == 0 and shares > 0:
                    # Exit Long to Cash
                    exec_price = price * (1.0 - self.slippage_pct)
                    proceeds = (shares * exec_price) - self.commission_per_trade
                    cash += max(0.0, proceeds)

                    # Record closed trade
                    trade_pnl = proceeds - (current_entry_shares * current_entry_price)
                    trade_return = (exec_price - current_entry_price) / current_entry_price
                    holding_days = (date - current_entry_date).days if current_entry_date else 0

                    trades.append(
                        Trade(
                            entry_date=current_entry_date,
                            entry_price=current_entry_price,
                            exit_date=date,
                            exit_price=exec_price,
                            position_type="LONG",
                            pnl=trade_pnl,
                            return_pct=trade_return,
                            holding_days=holding_days,
                        )
                    )

                    shares = 0.0
                    current_entry_date = None

            # Calculate total equity for today
            current_equity = cash + (shares * price)
            portfolio_equity.append(current_equity)

        data["Strategy_Equity"] = portfolio_equity

        # 4. Benchmark: 100% Buy & Hold from day 1
        initial_price = data["Close"].iloc[0]
        benchmark_shares = self.initial_capital / initial_price
        data["Benchmark_Equity"] = benchmark_shares * data["Close"]

        # 5. Calculate Metrics
        trade_returns = [t.return_pct for t in trades]
        metrics = calculate_metrics(
            equity_series=data["Strategy_Equity"],
            benchmark_equity=data["Benchmark_Equity"],
            trade_returns=trade_returns,
            risk_free_rate=self.risk_free_rate,
        )

        return BacktestResult(
            ticker=ticker,
            strategy_name=strategy.name,
            start_date=data.index[0].strftime("%Y-%m-%d"),
            end_date=data.index[-1].strftime("%Y-%m-%d"),
            initial_capital=self.initial_capital,
            final_equity=data["Strategy_Equity"].iloc[-1],
            benchmark_final_equity=data["Benchmark_Equity"].iloc[-1],
            metrics=metrics,
            equity_df=data,
            trades=trades,
        )
