"""
Risk & Execution Validator (Stage 3).

Checks simulated account holdings, cash sufficiency, position caps (linked with
R1-R4 risk analyzer), fundamental disclosure caveats (S08), and strict earliest
execution timing rules to prevent lookahead bias.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from .models import ForwardSignalReport, RiskValidationResult
from ..metrics import calculate_metrics
from ..risk_analyzer import assess_investment_risk


def validate_risk_and_execution(
    report: ForwardSignalReport,
    initial_capital: float = 10000.0,
    backtest_result: Optional[Any] = None,
    risk_profile: Optional[Dict[str, Any]] = None,
    price_series: Optional[pd.Series] = None,
) -> RiskValidationResult:
    """
    Validates whether a confirmed or conditional signal is executable under
    capital, cash buffer, position cap, and anti-lookahead rules.
    """
    warnings: List[str] = []

    # 1. Obtain or estimate Risk Profile (R1-R4)
    if risk_profile is None and price_series is not None and len(price_series.dropna()) >= 20:
        clean_px = price_series.dropna()
        eq = (initial_capital / float(clean_px.iloc[0])) * clean_px
        m = calculate_metrics(eq, benchmark_equity=eq)
        risk_profile = assess_investment_risk(m, ticker=report.ticker_or_pool)

    risk_level = risk_profile.get("risk_level", "未评估") if risk_profile else "未评估"
    risk_score = float(risk_profile["risk_score"]) if (risk_profile and "risk_score" in risk_profile) else None
    max_pos_hint = risk_profile.get("max_position", "按风控分散配置") if risk_profile else "按风控分散配置"
    stop_loss_hint = risk_profile.get("stop_loss", "按策略止损规则") if risk_profile else "按策略止损规则"

    # 2. Current position & cash check
    if not report.is_portfolio:
        if report.confirmed_position == 1:
            pos_state = "当前策略处于【持多仓】状态 (Position = 1)"
            cash_check = "持仓卖出无需额外现金，但需确认账户实际持有对应股数；卖出回笼现金最早于结算后可用。"
        else:
            pos_state = "当前策略处于【空仓观望】状态 (Position = 0)"
            if initial_capital >= report.latest_close:
                est_shares = int(initial_capital // report.latest_close)
                cash_check = f"模拟可用资金 ${initial_capital:,.2f} 充足（按最新价约可买入 {est_shares} 股，未含滑点）。"
            else:
                cash_check = f"⚠️ 模拟资金 ${initial_capital:,.2f} 低于 1 股最新价 ${report.latest_close:,.2f}（需支持碎股或提高本金）。"
                warnings.append("可用资金不足 1 整股价格，实盘下单需支持碎股交易。")

        # Position cap check: single-stock engine uses 100% capital -> warn if R3/R4!
        if risk_profile and risk_profile.get("badge") in {"HIGH_RISK", "EXTREME_RISK"}:
            pos_limit_check = (
                f"⚠️ 注意：单股回测引擎默认 100% 满仓，但该标的评级为【{risk_level}】，"
                f"风控建议单股仓位上限为【{max_pos_hint}】，防守止损红线【{stop_loss_hint}】。"
            )
            warnings.append(f"标的属于 {risk_level}，实盘切勿直接照搬回测 100% 单股满仓，建议控制在 {max_pos_hint}。")
        else:
            pos_limit_check = f"标的评级【{risk_level}】，建议单股仓位上限【{max_pos_hint}】，止损参考【{stop_loss_hint}】。"
    else:
        pos_state = f"多资产组合策略（当前已确认状态：{report.confirmed_signal}）"
        cash_pct = report.extra.get("cash_pct")
        min_cash_pct = report.extra.get("min_cash_pct")
        if cash_pct is not None and min_cash_pct is not None:
            if cash_pct <= min_cash_pct + 0.005:
                cash_check = f"⚠️ 当前模拟现金占比 {cash_pct*100:.2f}%，已贴近最低现金底线 {min_cash_pct*100:.0f}%，低配补仓将受现金约束限制。"
                warnings.append("组合现金已接近最低红线，补仓可能仅能部分执行。")
            else:
                cash_check = f"✅ 当前模拟现金占比 {cash_pct*100:.2f}%，高于最低现金底线 {min_cash_pct*100:.0f}%，具备再平衡缓冲。"
        else:
            cash_check = f"模拟初始资金 ${initial_capital:,.2f}；组合调仓严格遵循先卖后买、无融资杠杆原则。"

        if report.strategy_code == "S09":
            pos_limit_check = "S09 为单 ETF 100% 全仓宏观轮动模型（SPY/VEU/BIL），风险由绝对动量避险机制控制。"
        else:
            pos_limit_check = f"组合内置单股分散上限与现金缓冲，综合评级参考：【{risk_level}】。"

    # 3. Special Guardrails (e.g. S08 Fundamentals & Intraday wick dependency)
    if report.strategy_code == "S08":
        uncovered = report.extra.get("uncovered_tickers", [])
        guardrail = (
            "🛑 基本面财报边界提醒：S08 依赖季度财报披露（ROIC/FCF/杠杆率），不具备纯股价日内触发条件；"
            "请以最新官方财报与公告为准，避免在财报季前盲目假设静态指标。"
        )
        warnings.append("S08 不可仅凭股价预测下一日触发价，需核对真实财报披露数据。")
        if uncovered:
            warnings.append(f"资产池中有 {len(uncovered)} 只标的（如 {', '.join(uncovered[:4])}）未收录于本地基础财报快照，回测中已被跳过。")
    elif report.primary_trigger and report.primary_trigger.intraday_low_required:
        guardrail = (
            "⚠️ 盘中路径依赖提醒：S06 条件 B 需要盘中最低价触及 SMA50 且收盘重新站上触发线，无法仅凭单一收盘价完全判定。"
        )
        warnings.append("存在盘中最低价（Low）路径依赖，实盘需同时关注盘中回踩幅度与尾盘收盘位置。")
    else:
        guardrail = (
            f"✅ 策略风控与数据边界通过：风险评分 {f'{risk_score:.1f}/100' if risk_score is not None else 'N/A'}（{risk_level}）；"
            "若近期存在财报发布、拆股或重大宏观事件，建议实盘人工复核跳空与流动性风险。"
        )

    # 4. Anti-lookahead earliest execution rule
    if report.confirmed_is_new_event:
        exec_rule = (
            f"🟢 已确认信号（截至 {report.as_of_date} 收盘）→ 最早可执行时间：下一交易日 ({report.next_trading_date})。"
            "（无未来函数：收盘数据已完整锁定）"
        )
    else:
        exec_rule = (
            f"🔵 条件性信号（针对 {report.next_trading_date} 盘后判定）→ 若明日收盘触发，最早可执行时间：T+2 交易日。"
            "（严禁在 T+1 开盘提前假设触发，杜绝未来函数）"
        )

    # 5. Overall validator status
    if report.calculation_category == "FUNDAMENTAL_BOUNDED":
        status = "CAUTION"
        passed = True
    elif any("不足 1 整股" in w for w in warnings):
        status = "BLOCKED"
        passed = False
    elif warnings:
        status = "CAUTION"
        passed = True
    else:
        status = "EXECUTABLE"
        passed = True

    return RiskValidationResult(
        passed=passed,
        status=status,
        current_position_state=pos_state,
        cash_check=cash_check,
        position_limit_check=pos_limit_check,
        risk_guardrail_check=guardrail,
        earliest_execution_rule=exec_rule,
        risk_level=risk_level,
        risk_score=risk_score,
        warnings=warnings,
    )
