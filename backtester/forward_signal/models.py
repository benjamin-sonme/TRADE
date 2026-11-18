"""
Data models for the Next-Day Signal Preview (Forward Signal Analysis) system.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


@dataclass
class TriggerCondition:
    """Describes the next-trading-day trigger condition for BUY, SELL, or REBALANCE."""

    action: str  # "BUY" | "SELL" | "REBALANCE"
    can_trigger_next_day: bool
    trigger_price: Optional[float] = None
    price_range: Optional[Tuple[Optional[float], Optional[float]]] = None  # (low_bound, high_bound)
    comparison: str = "BLOCKED"  # ">" | ">=" | "<" | "<=" | "RANGE" | "SCENARIO" | "BLOCKED"
    distance_pct: Optional[float] = None  # (trigger_price - latest_close) / latest_close
    distance_abs: Optional[float] = None  # trigger_price - latest_close
    condition_summary: str = ""
    blocker_reason: Optional[str] = None
    earliest_execution: str = ""
    intraday_low_required: bool = False
    intraday_low_threshold: Optional[float] = None

    def format_trigger_label(self) -> str:
        """Human-readable compact price/condition label for KPI cards."""
        if not self.can_trigger_next_day:
            return "当前无法触发"
        if self.comparison == "RANGE" and self.price_range is not None:
            lo, hi = self.price_range
            if lo is not None and hi is not None:
                return f"${lo:,.2f} < 收盘价 ≤ ${hi:,.2f}"
            if lo is not None:
                return f"收盘价 > ${lo:,.2f}"
            if hi is not None:
                return f"收盘价 ≤ ${hi:,.2f}"
        if self.trigger_price is not None:
            op_map = {">": ">", ">=": "≥", "<": "<", "<=": "≤"}
            op = op_map.get(self.comparison, "")
            suffix = " (且盘中触及均线)" if self.intraday_low_required else ""
            return f"{op} ${self.trigger_price:,.2f}{suffix}"
        return "见多资产情景条件"


@dataclass
class RiskValidationResult:
    """Output of the Risk & Execution Validator module."""

    passed: bool
    status: str  # "EXECUTABLE" | "CAUTION" | "BLOCKED"
    current_position_state: str
    cash_check: str
    position_limit_check: str
    risk_guardrail_check: str
    earliest_execution_rule: str
    risk_level: str = ""
    risk_score: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class ForwardSignalReport:
    """Unified Next-Day Signal Preview report for any strategy S01-S09."""

    strategy_code: str
    strategy_name: str
    strategy_en_name: str
    is_portfolio: bool
    ticker_or_pool: str
    as_of_date: str
    next_trading_date: str
    latest_close: float

    # Classification (Three categories)
    calculation_category: str  # "EXACT_THRESHOLD" | "DYNAMIC_SCENARIO" | "FUNDAMENTAL_BOUNDED"
    calculation_category_label: str

    # Tier 1: Confirmed Signal (Latest completed trading day T)
    confirmed_signal: str  # "BUY" | "SELL" | "HOLD" | "WAIT" | "REBALANCE"
    confirmed_is_new_event: bool
    confirmed_position: int  # 1 (long/invested) or 0 (flat/cash)
    confirmed_summary: str
    confirmed_execution_timing: str

    # Tier 2: Conditional Signal (Next trading day T+1)
    conditional_summary: str
    conditional_execution_timing: str

    # Core 5 Fields
    current_signal: str  # "BUY" | "SELL" | "HOLD" | "WAIT" | "REBALANCE"
    next_buy_trigger: TriggerCondition
    next_sell_trigger: TriggerCondition
    primary_trigger: Optional[TriggerCondition]
    execution_timing: str

    # Why This Signal?
    why_items: List[Dict[str, str]] = field(default_factory=list)
    why_explanation: List[str] = field(default_factory=list)
    next_rebalance_date: Optional[str] = None
    scenario_table: Optional[pd.DataFrame] = None

    # Risk & Execution Validator
    risk_validation: Optional[RiskValidationResult] = None

    # Extra diagnostics
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SensitivityRow:
    """Single hypothetical next-day closing price scenario."""

    test_price: float
    pct_change: float
    resulting_signal: str  # "BUY" | "SELL" | "HOLD" | "WAIT"
    raw_position: int
    is_transition: bool
    indicator_snapshot: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplayValidationResult:
    """Output of Historical Replay Validation (strict no-lookahead audit)."""

    strategy_code: str
    ticker_or_pool: str
    total_days_tested: int
    scenario_exact_matches: int
    threshold_exact_matches: int
    mismatches: int
    scenario_accuracy_pct: float
    threshold_accuracy_pct: float
    buy_transitions_actual: int
    buy_transitions_matched: int
    sell_transitions_actual: int
    sell_transitions_matched: int
    details_df: pd.DataFrame
    notes: List[str] = field(default_factory=list)
