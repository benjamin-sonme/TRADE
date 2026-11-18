"""
Next-Day Signal Preview (Forward Signal Analysis) Package.

Architecture (4 Core Modules):
1. Historical Data & Strategy Engine -> canonical strategy factories in `scenario_engine.py`
2. Forward Scenario Engine           -> `ForwardScenarioEngine` in `scenario_engine.py`
3. Signal Threshold Solver           -> `solve_single_stock_forward_signal` & `solve_portfolio_forward_signal`
4. Risk & Execution Validator        -> `validate_risk_and_execution`

Extra Verification & Decision-Support Tools:
- Historical Replay Validation       -> `run_historical_replay_validation` / `run_portfolio_replay_validation`
- Price Sensitivity Analysis         -> `run_price_sensitivity` / `run_cross_strategy_sensitivity`
"""

from typing import Any, Dict, List, Optional
import pandas as pd

from .models import (
    ForwardSignalReport,
    ReplayValidationResult,
    RiskValidationResult,
    SensitivityRow,
    TriggerCondition,
)
from .scenario_engine import (
    STRATEGY_METADATA,
    ForwardScenarioEngine,
    classify_transition,
    create_portfolio_strategy,
    create_single_stock_strategy,
    estimate_next_trading_date,
    is_month_end_business_day,
    next_month_end_business_day,
)
from .threshold_solver import solve_single_stock_forward_signal
from .portfolio_solver import solve_portfolio_forward_signal
from .risk_validator import validate_risk_and_execution
from .analysis_tools import (
    generate_default_price_grid,
    parse_custom_prices,
    run_cross_strategy_sensitivity,
    run_historical_replay_validation,
    run_portfolio_replay_validation,
    run_price_sensitivity,
)


def analyze_next_day_signal(
    strategy_code: str,
    df: Optional[pd.DataFrame] = None,
    prices_df: Optional[pd.DataFrame] = None,
    ticker: str = "NVDA",
    universe: Optional[List[str]] = None,
    initial_capital: float = 10000.0,
    slippage_bps: float = 5.0,
    eval_days: Optional[int] = None,
    backtest_result: Optional[Any] = None,
    risk_profile: Optional[Dict[str, Any]] = None,
) -> ForwardSignalReport:
    """
    Unified entry point for Next-Day Signal Preview across all 9 strategies (S01-S09).
    Executes Scenario Engine -> Threshold/Portfolio Solver -> Risk & Execution Validator.
    """
    code = strategy_code.upper()
    meta = STRATEGY_METADATA[code]

    if not meta["is_portfolio"]:
        if df is None or df.empty:
            raise ValueError(f"Strategy {code} requires a non-empty single-stock OHLCV DataFrame.")
        report = solve_single_stock_forward_signal(strategy_code=code, df=df, ticker=ticker)
        px_ser = df["Close"]
    else:
        if prices_df is None or prices_df.empty:
            raise ValueError(f"Strategy {code} requires a non-empty multi-asset prices_df.")
        u = list(universe) if universe else list(prices_df.columns)
        report = solve_portfolio_forward_signal(
            strategy_code=code,
            prices_df=prices_df,
            universe=u,
            focus_ticker=ticker if ticker in prices_df.columns else None,
            initial_capital=initial_capital,
            slippage_bps=slippage_bps,
            eval_days=eval_days,
        )
        px_ser = prices_df[ticker] if ticker in prices_df.columns else prices_df.iloc[:, 0]

    report.risk_validation = validate_risk_and_execution(
        report=report,
        initial_capital=initial_capital,
        backtest_result=backtest_result,
        risk_profile=risk_profile,
        price_series=px_ser,
    )
    return report


__all__ = [
    "STRATEGY_METADATA",
    "TriggerCondition",
    "RiskValidationResult",
    "ForwardSignalReport",
    "SensitivityRow",
    "ReplayValidationResult",
    "ForwardScenarioEngine",
    "create_single_stock_strategy",
    "create_portfolio_strategy",
    "estimate_next_trading_date",
    "is_month_end_business_day",
    "next_month_end_business_day",
    "classify_transition",
    "solve_single_stock_forward_signal",
    "solve_portfolio_forward_signal",
    "validate_risk_and_execution",
    "analyze_next_day_signal",
    "generate_default_price_grid",
    "parse_custom_prices",
    "run_price_sensitivity",
    "run_cross_strategy_sensitivity",
    "run_historical_replay_validation",
    "run_portfolio_replay_validation",
]
