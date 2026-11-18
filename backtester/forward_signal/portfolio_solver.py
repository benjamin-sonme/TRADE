"""
Forward Signal Solver for Portfolio & Multi-Asset Strategies:
- Stage 2: S04 (Relative Momentum), S07 (Threshold Rebalancing), S09 (Dual Momentum)
- Stage 3: S08 (Quality & Value Factor - fundamental-bounded, no fabricated price triggers)

Calls the exact strategy implementations (`compute_target_weights` / `simulate`)
as the single source of truth.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from .models import ForwardSignalReport, TriggerCondition
from .scenario_engine import (
    STRATEGY_METADATA,
    create_portfolio_strategy,
    estimate_next_trading_date,
    is_month_end_business_day,
    next_month_end_business_day,
)
from ..strategies.quality_value_factor import DEFAULT_FUNDAMENTALS_DATABASE

EXEC_PORTFOLIO_CONFIRMED = "最新交易日为调仓检验日且已触发调仓 · 最早于下一交易日 (T+1) 执行再平衡"
EXEC_PORTFOLIO_NEXT_DAY_COND = "下一交易日 (T+1) 为调仓检验日 · 若 T+1 收盘触发调仓，最早于 T+2 交易日执行"


def solve_portfolio_forward_signal(
    strategy_code: str,
    prices_df: pd.DataFrame,
    universe: List[str],
    focus_ticker: Optional[str] = None,
    initial_capital: float = 100000.0,
    slippage_bps: float = 5.0,
    eval_days: Optional[int] = None,
) -> ForwardSignalReport:
    """Computes Next-Day Signal Preview for S04, S07, S08, or S09."""
    code = strategy_code.upper()
    if prices_df is None or prices_df.empty:
        raise ValueError("Multi-asset prices_df is empty.")

    if code == "S04":
        return _solve_s04(prices_df, universe, focus_ticker)
    if code == "S07":
        return _solve_s07(prices_df, universe, focus_ticker, initial_capital, slippage_bps, eval_days)
    if code == "S08":
        return _solve_s08(prices_df, universe, focus_ticker)
    if code == "S09":
        return _solve_s09(prices_df, focus_ticker)
    raise ValueError(f"Unsupported portfolio strategy code: {strategy_code}")


def _solve_s04(
    prices_df: pd.DataFrame,
    universe: List[str],
    focus_ticker: Optional[str] = None,
) -> ForwardSignalReport:
    meta = STRATEGY_METADATA["S04"]
    strategy = create_portfolio_strategy("S04", universe)
    weights_df = strategy.compute_target_weights(prices_df)

    as_of_ts = pd.Timestamp(prices_df.index[-1])
    next_ts = estimate_next_trading_date(as_of_ts)
    today_is_me = is_month_end_business_day(as_of_ts)
    tomorrow_is_me = is_month_end_business_day(next_ts)
    next_rebal_ts = next_month_end_business_day(as_of_ts, include_today=False)

    as_of_str = as_of_ts.strftime("%Y-%m-%d")
    next_str = next_ts.strftime("%Y-%m-%d")
    next_rebal_str = next_rebal_ts.strftime("%Y-%m-%d")

    focus = focus_ticker if (focus_ticker and focus_ticker in prices_df.columns) else str(prices_df.columns[0])
    latest_close = float(prices_df[focus].dropna().iloc[-1])

    # Compute exact 12-1 momentum scores as of latest date using the exact same formula as RelativeMomentumRotation
    rows = []
    for asset in prices_df.columns:
        series = prices_df[asset].dropna()
        if as_of_ts not in series.index:
            rows.append({"标的": asset, "最新价": np.nan, "12-1 动量": np.nan, "数据状态": "最新日缺失"})
            continue
        a_loc = series.index.get_loc(as_of_ts)
        px_now = float(series.iloc[a_loc])
        score = np.nan
        note = "历史不足"
        if a_loc >= strategy.lookback_days:
            p_21 = series.iloc[a_loc - strategy.skip_recent_days]
            p_252 = series.iloc[a_loc - strategy.lookback_days]
            if pd.notna(p_21) and pd.notna(p_252) and p_252 > 0:
                score = float((p_21 / p_252) - 1.0)
                note = "完整 12-1 动量"
        elif a_loc >= strategy.skip_recent_days + 40:
            p_21 = series.iloc[a_loc - strategy.skip_recent_days]
            p_start = series.iloc[0]
            if pd.notna(p_21) and pd.notna(p_start) and p_start > 0:
                score = float((p_21 / p_start) - 1.0)
                note = "次新自适应动量"
        rows.append({"标的": asset, "最新价": px_now, "12-1 动量": score, "数据状态": note})

    rank_df = pd.DataFrame(rows).sort_values("12-1 动量", ascending=False, na_position="last").reset_index(drop=True)
    rank_df["排名"] = np.arange(1, len(rank_df) + 1)

    # Determine confirmed active weights (from last COMPLETED month-end) vs pro-forma latest weights
    completed_weights = weights_df[weights_df.index < as_of_ts] if not today_is_me else weights_df
    active_w = completed_weights.iloc[-1].to_dict() if not completed_weights.empty else {a: 0.0 for a in prices_df.columns}
    proforma_w = weights_df.iloc[-1].to_dict() if not weights_df.empty else active_w

    active_selected = [a for a, w in active_w.items() if w > 0]
    proforma_selected = [a for a, w in proforma_w.items() if w > 0]

    rank_df["当前已确认权重"] = rank_df["标的"].map(lambda a: f"{active_w.get(a, 0.0)*100:.0f}%")
    rank_df["当前排位目标权重"] = rank_df["标的"].map(lambda a: f"{proforma_w.get(a, 0.0)*100:.0f}%")
    rank_df["若今日调仓动作"] = rank_df["标的"].map(
        lambda a: (
            "BUY (新入选)" if proforma_w.get(a, 0.0) > active_w.get(a, 0.0)
            else ("SELL (拟剔除)" if proforma_w.get(a, 0.0) < active_w.get(a, 0.0)
                  else ("HOLD (续持)" if active_w.get(a, 0.0) > 0 else "WAIT"))
        )
    )
    rank_df["最新价"] = rank_df["最新价"].map(lambda v: f"${v:,.2f}" if pd.notna(v) else "N/A")
    rank_df["12-1 动量"] = rank_df["12-1 动量"].map(lambda v: f"{v*100:+.2f}%" if pd.notna(v) else "N/A")
    rank_df = rank_df[["排名", "标的", "最新价", "12-1 动量", "当前已确认权重", "当前排位目标权重", "若今日调仓动作", "数据状态"]]

    # Confirmed signal status
    weight_changed = set(active_selected) != set(proforma_selected)
    if today_is_me and weight_changed:
        confirmed_sig = "REBALANCE"
        confirmed_is_new = True
        confirmed_summary = (
            f"今日 ({as_of_str}) 为月末调仓日，已确认新组合目标：{', '.join(proforma_selected) or '100% 现金'}（每只 20%）。"
        )
        confirmed_exec = EXEC_PORTFOLIO_CONFIRMED
    else:
        confirmed_sig = "HOLD" if active_selected else "WAIT"
        confirmed_is_new = False
        confirmed_summary = (
            f"截至 {as_of_str} 非月末调仓日，维持上月末已确认持仓：{', '.join(active_selected) or '全现金观望'}。"
        )
        confirmed_exec = f"非月末调仓日 · 维持现有持仓，下一调仓检验日为 {next_rebal_str}"

    # Next-day buy/sell triggers (multi-asset + calendar gated)
    if tomorrow_is_me:
        to_buy = [a for a in proforma_selected if a not in active_selected]
        to_sell = [a for a in active_selected if a not in proforma_selected]
        buy_trig = TriggerCondition(
            action="BUY",
            can_trigger_next_day=bool(to_buy),
            comparison="SCENARIO",
            condition_summary=(
                f"明日 ({next_str}) 为月末调仓日！因 12-1 动量跳过近 21 日价格，按历史锁定动量预计新买入：{', '.join(to_buy)}（各配 20%）。"
                if to_buy
                else f"明日 ({next_str}) 虽为月末调仓日，但按当前 12-1 动量排名无新增入选标的。"
            ),
            blocker_reason=None if to_buy else "当前前 4 名正动量标的与现有持仓一致，预计无需新买入。",
            earliest_execution=EXEC_PORTFOLIO_NEXT_DAY_COND,
        )
        sell_trig = TriggerCondition(
            action="SELL",
            can_trigger_next_day=bool(to_sell),
            comparison="SCENARIO",
            condition_summary=(
                f"明日 ({next_str}) 为月末调仓日，预计剔除卖出：{', '.join(to_sell)}。"
                if to_sell
                else f"明日 ({next_str}) 为月末调仓日，现有持仓均保住前 {strategy.top_n} 正动量席位，预计无需卖出。"
            ),
            blocker_reason=None if to_sell else "现有持仓仍位于 Top 4 正动量区间。",
            earliest_execution=EXEC_PORTFOLIO_NEXT_DAY_COND,
        )
        cond_summary = f"明日 ({next_str}) 即为月末调仓检验日！系统将按 12-1 动量选取正动量 Top {strategy.top_n}（拟买入：{', '.join(to_buy) or '无'}；拟卖出：{', '.join(to_sell) or '无'}）。"
        cond_exec = EXEC_PORTFOLIO_NEXT_DAY_COND
    else:
        buy_trig = TriggerCondition(
            action="BUY",
            can_trigger_next_day=False,
            comparison="BLOCKED",
            condition_summary=f"明日 ({next_str}) 非月末调仓日，不会触发换仓买入",
            blocker_reason=f"S04 仅在每月最后一个交易日调仓，下一个调仓检验日为 {next_rebal_str}（届时将根据 12-1 动量排名前 {strategy.top_n} 且 M_t > 0 选股）。",
        )
        sell_trig = TriggerCondition(
            action="SELL",
            can_trigger_next_day=False,
            comparison="BLOCKED",
            condition_summary=f"明日 ({next_str}) 非月末调仓日，不会触发换仓卖出",
            blocker_reason=f"须等待至下一月末检验日 ({next_rebal_str})；若持仓跌出前 {strategy.top_n} 或动量转负 (M_t ≤ 0) 则触发卖出。",
        )
        cond_summary = f"明日 ({next_str}) 非月末调仓日，不产生交易信号；下一调仓检验日为 {next_rebal_str}。"
        cond_exec = f"最早调仓检验日：{next_rebal_str} 收盘确认 → 次月首个交易日执行"

    primary = buy_trig if (tomorrow_is_me and buy_trig.can_trigger_next_day) else (sell_trig if (tomorrow_is_me and sell_trig.can_trigger_next_day) else buy_trig)

    why_items = [
        {"指标": "当前已确认持仓", "当前值": ", ".join(active_selected) or "100% 现金", "说明": f"每只目标权重 {int(strategy.target_weight_per_stock*100)}%，保留 ≥ {int(strategy.min_cash_pct*100)}% 现金"},
        {"指标": "当前动态排名 Top 4", "当前值": ", ".join(proforma_selected) or "无正动量标的", "说明": "按最新截面 12-1 动量排序"},
        {"指标": "下一调仓检验日", "当前值": next_rebal_str, "说明": "月末最后一个交易日收盘计算"},
        {"指标": "股票池数量", "当前值": f"{len(prices_df.columns)} 只", "说明": "多资产相对排名决定，非单股固定价格触发"},
    ]
    why_expl = [
        "S04 为**横截面相对动量策略**：无法用单只股票的固定价格决定买卖，而是依赖整个股票池的 `12-1` 动量相对排名（`P_{t-21}/P_{t-252} - 1`）与月末调仓日历。",
        f"由于 `12-1` 动量跳过了最近 21 个交易日，临近月末时排名主要由 21 个交易日前的历史价格锁定，抗短线噪音干扰。",
        f"下一个月末检验日为 **{next_rebal_str}**；确认后的调仓单最早于下月首个交易日执行。",
    ]

    return ForwardSignalReport(
        strategy_code="S04",
        strategy_name=f"S04 - {meta['cn_name']}",
        strategy_en_name=meta["en_name"],
        is_portfolio=True,
        ticker_or_pool=f"组合池 ({len(prices_df.columns)} 只 · 焦点 {focus})",
        as_of_date=as_of_str,
        next_trading_date=next_str,
        latest_close=latest_close,
        calculation_category=meta["category"],
        calculation_category_label=meta["category_label"],
        confirmed_signal=confirmed_sig,
        confirmed_is_new_event=confirmed_is_new,
        confirmed_position=1 if active_selected else 0,
        confirmed_summary=confirmed_summary,
        confirmed_execution_timing=confirmed_exec,
        conditional_summary=cond_summary,
        conditional_execution_timing=cond_exec,
        current_signal=confirmed_sig,
        next_buy_trigger=buy_trig,
        next_sell_trigger=sell_trig,
        primary_trigger=primary,
        execution_timing=confirmed_exec if confirmed_is_new else cond_exec,
        why_items=why_items,
        why_explanation=why_expl,
        next_rebalance_date=next_rebal_str,
        scenario_table=rank_df,
    )


def _solve_s07(
    prices_df: pd.DataFrame,
    universe: List[str],
    focus_ticker: Optional[str],
    initial_capital: float,
    slippage_bps: float,
    eval_days: Optional[int],
) -> ForwardSignalReport:
    meta = STRATEGY_METADATA["S07"]
    strategy = create_portfolio_strategy("S07", universe)
    res = strategy.simulate(
        prices_df=prices_df,
        initial_capital=initial_capital,
        slippage_bps=slippage_bps,
        eval_days=eval_days,
    )

    as_of_ts = pd.Timestamp(res.equity_df.index[-1])
    next_ts = estimate_next_trading_date(as_of_ts)
    today_is_me = is_month_end_business_day(as_of_ts)
    tomorrow_is_me = is_month_end_business_day(next_ts)
    next_rebal_ts = next_month_end_business_day(as_of_ts, include_today=False)

    as_of_str = as_of_ts.strftime("%Y-%m-%d")
    next_str = next_ts.strftime("%Y-%m-%d")
    next_rebal_str = next_rebal_ts.strftime("%Y-%m-%d")

    latest_weights = res.weights_history.iloc[-1].to_dict()
    total_equity = float(res.final_equity)
    cash_w = float(latest_weights.get("CASH", 0.0))

    focus = focus_ticker if (focus_ticker and focus_ticker in prices_df.columns) else str(prices_df.columns[0])
    latest_close = float(prices_df[focus].dropna().iloc[-1])

    band_rows = []
    breached_high = []
    breached_low = []
    focus_trim_px: Optional[float] = None
    focus_add_px: Optional[float] = None

    for asset, band in strategy.bands.items():
        w = float(latest_weights.get(asset, 0.0))
        px_ser = prices_df[asset].dropna()
        px = float(px_ser.iloc[-1]) if not px_ser.empty else np.nan
        upper = band["upper"]
        lower = band["lower"]
        target = band["target"]

        # Ceteris-paribus single-asset price that breaches upper / lower tolerance band:
        # w(P) = (q * P) / (q * P + V_other) => P_thresh = thresh * V_other / (q * (1 - thresh))
        trim_px = np.nan
        add_px = np.nan
        if pd.notna(px) and px > 0 and w > 0:
            v_i = total_equity * w
            v_other = max(0.0, total_equity - v_i)
            q_i = v_i / px
            if q_i > 0 and upper < 1.0:
                trim_px = (upper * v_other) / (q_i * (1.0 - upper))
            if q_i > 0 and lower > 0.0 and lower < 1.0:
                add_px = (lower * v_other) / (q_i * (1.0 - lower))

        if asset == focus:
            focus_trim_px = float(trim_px) if pd.notna(trim_px) else None
            focus_add_px = float(add_px) if pd.notna(add_px) else None

        if w > upper:
            status = "🔴 超出上限 (拟减仓)"
            breached_high.append(asset)
        elif w < lower:
            status = "🟢 低于下限 (拟补仓)"
            breached_low.append(asset)
        else:
            status = "⚪ 容忍带内 (不动)"

        band_rows.append(
            {
                "标的": asset,
                "最新价": f"${px:,.2f}" if pd.notna(px) else "N/A",
                "当前权重": f"{w*100:.2f}%",
                "目标权重": f"{target*100:.2f}%",
                "容忍区间": f"[{lower*100:.1f}%, {upper*100:.1f}%]",
                "静态减仓临界价*": f">${trim_px:,.2f}" if pd.notna(trim_px) else "N/A",
                "静态补仓临界价*": f"<${add_px:,.2f}" if pd.notna(add_px) else "N/A",
                "当前状态": status,
            }
        )

    scenario_df = pd.DataFrame(band_rows)
    any_breach = bool(breached_high or breached_low)

    if today_is_me and any_breach:
        confirmed_sig = "REBALANCE"
        confirmed_is_new = True
        confirmed_summary = (
            f"今日 ({as_of_str}) 为月末检验日且权重已越界：超配拟减仓 [{', '.join(breached_high) or '无'}]，"
            f"低配拟补仓 [{', '.join(breached_low) or '无'}]。"
        )
        confirmed_exec = EXEC_PORTFOLIO_CONFIRMED
    else:
        confirmed_sig = "HOLD"
        confirmed_is_new = False
        confirmed_summary = (
            f"截至 {as_of_str} 组合净值 ${total_equity:,.2f}（现金占比 {cash_w*100:.1f}%），"
            + (
                f"已有标的偏离容忍带（超配: {', '.join(breached_high) or '无'}；低配: {', '.join(breached_low) or '无'}），但非月末检验日暂不调仓。"
                if any_breach
                else "所有成分股权重均处于容忍带内，维持零摩擦持有。"
            )
        )
        confirmed_exec = f"非月末检验日 · 维持当前仓位，下一检验日为 {next_rebal_str}"

    if tomorrow_is_me:
        buy_trig = TriggerCondition(
            action="BUY",
            can_trigger_next_day=True,
            trigger_price=focus_add_px,
            comparison="<" if focus_add_px else "SCENARIO",
            distance_pct=((focus_add_px - latest_close) / latest_close) if (focus_add_px and latest_close > 0) else None,
            distance_abs=(focus_add_px - latest_close) if focus_add_px else None,
            condition_summary=(
                f"明日 ({next_str}) 为月末检验日！任意标的权重跌破下限（且现金 > {strategy.min_cash_pct*100:.0f}% 底线）将触发补仓；"
                f"其余资产不变情景下，焦点股 {focus} 跌破 ${focus_add_px:,.2f} 将触发补仓。"
                if focus_add_px
                else f"明日 ({next_str}) 为月末检验日，当任一成分股权重低于其下限且现金高于底线时触发补仓。"
            ),
            earliest_execution=EXEC_PORTFOLIO_NEXT_DAY_COND,
        )
        sell_trig = TriggerCondition(
            action="SELL",
            can_trigger_next_day=True,
            trigger_price=focus_trim_px,
            comparison=">" if focus_trim_px else "SCENARIO",
            distance_pct=((focus_trim_px - latest_close) / latest_close) if (focus_trim_px and latest_close > 0) else None,
            distance_abs=(focus_trim_px - latest_close) if focus_trim_px else None,
            condition_summary=(
                f"明日 ({next_str}) 为月末检验日！任意标的权重突破上限将触发止盈减仓；"
                f"其余资产不变情景下，焦点股 {focus} 涨破 ${focus_trim_px:,.2f} 将触发减仓。"
                if focus_trim_px
                else f"明日 ({next_str}) 为月末检验日，当任一成分股权重高于其上限时触发减仓。"
            ),
            earliest_execution=EXEC_PORTFOLIO_NEXT_DAY_COND,
        )
        cond_summary = f"明日 ({next_str}) 即为月末权重偏移检验日，若任一标的越出 ±{int(strategy.band_pct*100)}% 容忍带，将于次日执行再平衡。"
        cond_exec = EXEC_PORTFOLIO_NEXT_DAY_COND
    else:
        buy_trig = TriggerCondition(
            action="BUY",
            can_trigger_next_day=False,
            comparison="BLOCKED",
            condition_summary=f"明日 ({next_str}) 非月末检验日，不触发再平衡补仓",
            blocker_reason=f"S07 仅在每月最后一个交易日检测权重偏离，下一检验日为 {next_rebal_str}；下表给出各标的在‘其余资产不变’情景下的临界价参考。",
        )
        sell_trig = TriggerCondition(
            action="SELL",
            can_trigger_next_day=False,
            comparison="BLOCKED",
            condition_summary=f"明日 ({next_str}) 非月末检验日，不触发再平衡减仓",
            blocker_reason=f"须等待至下一月末检验日 ({next_rebal_str})；届时任意资产权重突破上限才触发止盈减仓。",
        )
        cond_summary = f"明日 ({next_str}) 非月末检验日，预计零交易；下一检验日为 {next_rebal_str}。"
        cond_exec = f"最早检验日：{next_rebal_str} 收盘确认 → 次月首个交易日执行"

    why_items = [
        {"指标": "当前模拟组合净值", "当前值": f"${total_equity:,.2f}", "说明": f"初始本金 ${initial_capital:,.0f}"},
        {"指标": "当前现金占比", "当前值": f"{cash_w*100:.2f}%", "说明": f"目标 {strategy.target_cash_pct*100:.0f}% · 最低红线 {strategy.min_cash_pct*100:.0f}%"},
        {"指标": "越界标的数量", "当前值": f"{len(breached_high) + len(breached_low)} 只", "说明": f"超配 {len(breached_high)} 只 / 低配 {len(breached_low)} 只"},
        {"指标": "下一月末检验日", "当前值": next_rebal_str, "说明": "仅月末收盘检验，盘中不频繁换手"},
    ]
    why_expl = [
        "S07 权重由**整体组合市值与所有成分股共同涨跌**决定，因此单股临界价标注为“*其余成分股价格不变情景下的参考临界价”。",
        f"规则：单标的容忍带为 `max(2%, 25% × 目标权重)`；先减超配资产回笼现金，再在保持现金 ≥ `{strategy.min_cash_pct*100:.0f}%` 前提下补足低配资产。",
        f"当前现金占比 **{cash_w*100:.2f}%**，{'满足补仓现金底线' if cash_w > strategy.min_cash_pct else '已接近或触及现金底线，补仓将受限'}。",
    ]

    return ForwardSignalReport(
        strategy_code="S07",
        strategy_name=f"S07 - {meta['cn_name']}",
        strategy_en_name=meta["en_name"],
        is_portfolio=True,
        ticker_or_pool=f"再平衡组合 ({len(strategy.target_weights)} 只 · 焦点 {focus})",
        as_of_date=as_of_str,
        next_trading_date=next_str,
        latest_close=latest_close,
        calculation_category=meta["category"],
        calculation_category_label=meta["category_label"],
        confirmed_signal=confirmed_sig,
        confirmed_is_new_event=confirmed_is_new,
        confirmed_position=1,
        confirmed_summary=confirmed_summary,
        confirmed_execution_timing=confirmed_exec,
        conditional_summary=cond_summary,
        conditional_execution_timing=cond_exec,
        current_signal=confirmed_sig,
        next_buy_trigger=buy_trig,
        next_sell_trigger=sell_trig,
        primary_trigger=sell_trig if (tomorrow_is_me and breached_high) else buy_trig,
        execution_timing=confirmed_exec if confirmed_is_new else cond_exec,
        why_items=why_items,
        why_explanation=why_expl,
        next_rebalance_date=next_rebal_str,
        scenario_table=scenario_df,
        extra={"cash_pct": cash_w, "min_cash_pct": strategy.min_cash_pct, "total_equity": total_equity},
    )


def _solve_s08(
    prices_df: pd.DataFrame,
    universe: List[str],
    focus_ticker: Optional[str] = None,
) -> ForwardSignalReport:
    meta = STRATEGY_METADATA["S08"]
    strategy = create_portfolio_strategy("S08", universe)
    weights_df = strategy.compute_target_weights(prices_df)

    as_of_ts = pd.Timestamp(prices_df.index[-1])
    next_ts = estimate_next_trading_date(as_of_ts)
    as_of_str = as_of_ts.strftime("%Y-%m-%d")
    next_str = next_ts.strftime("%Y-%m-%d")

    n_days = len(prices_df)
    q_days = strategy.rebalance_quarter_days
    days_since_cycle = (n_days - 1) % q_days
    days_to_next = q_days - days_since_cycle if days_since_cycle != 0 else q_days
    next_rebal_ts = as_of_ts + pd.tseries.offsets.BDay(days_to_next)
    next_rebal_str = next_rebal_ts.strftime("%Y-%m-%d")

    focus = focus_ticker if (focus_ticker and focus_ticker in prices_df.columns) else str(prices_df.columns[0])
    latest_close = float(prices_df[focus].dropna().iloc[-1])

    latest_w = weights_df.iloc[-1].to_dict() if not weights_df.empty else {}
    selected = [a for a, w in latest_w.items() if w > 0]

    # Build transparent fundamental factor table (explicitly showing data coverage & snapshot limitations)
    factor_rows = []
    uncovered = []
    for asset in prices_df.columns:
        fund = DEFAULT_FUNDAMENTALS_DATABASE.get(asset)
        px_ser = prices_df[asset].dropna()
        px_now = float(px_ser.iloc[-1]) if not px_ser.empty else np.nan
        if not fund:
            uncovered.append(asset)
            factor_rows.append(
                {
                    "标的": asset,
                    "最新价": f"${px_now:,.2f}" if pd.notna(px_now) else "N/A",
                    "ROIC": "缺财报数据",
                    "净负债/EBITDA": "缺财报数据",
                    "估算 FCF Yield": "未覆盖",
                    "质量筛选": "⚠️ 未收录",
                    "当前目标权重": "0%",
                }
            )
            continue
        p0 = float(prices_df[asset].iloc[0]) if pd.notna(prices_df[asset].iloc[0]) and prices_df[asset].iloc[0] > 0 else px_now
        price_norm = (px_now / p0) if (pd.notna(px_now) and p0 > 0) else 1.0
        dyn_yield = fund["base_fcf_yield"] / price_norm if price_norm > 0 else fund["base_fcf_yield"]
        pass_q = (fund["roic"] >= strategy.min_roic) and (fund["fcf"] > 0) and (fund["net_debt_ebitda"] <= strategy.max_net_debt_ebitda)
        factor_rows.append(
            {
                "标的": asset,
                "最新价": f"${px_now:,.2f}" if pd.notna(px_now) else "N/A",
                "ROIC": f"{fund['roic']*100:.1f}%",
                "净负债/EBITDA": f"{fund['net_debt_ebitda']:.2f}x",
                "估算 FCF Yield": f"{dyn_yield*100:.2f}%",
                "质量筛选": "✅ 通过" if pass_q else "❌ 未通过",
                "当前目标权重": f"{latest_w.get(asset, 0.0)*100:.0f}%",
            }
        )

    scenario_df = pd.DataFrame(factor_rows)

    confirmed_sig = "HOLD" if selected else "WAIT"
    confirmed_summary = (
        f"截至 {as_of_str}，当前已确认基本面精选持仓：{', '.join(selected) or '无符合标的'}（各 15%，保留 25% 现金）。"
    )
    confirmed_exec = f"季频基本面再平衡 · 距下一计划检验日 ({next_rebal_str}) 约 {days_to_next} 个交易日"

    # Strictly refuse to fabricate single-stock price triggers for fundamental strategy!
    buy_trig = TriggerCondition(
        action="BUY",
        can_trigger_next_day=False,
        comparison="BLOCKED",
        condition_summary="不可仅凭下一交易日股价推算精确买入触发价（基本面财报主导）",
        blocker_reason=(
            "S08 依赖季度财报披露的 ROIC (≥10%)、TTM 自由现金流 (>0)、净负债/EBITDA (≤3.0) 以及全池 FCF Yield 横截面排序。"
            "未来财报与同行估值变动不可提前预知，因此系统不编造虚假的精确价格触发点。"
        ),
    )
    sell_trig = TriggerCondition(
        action="SELL",
        can_trigger_next_day=False,
        comparison="BLOCKED",
        condition_summary="不可仅凭下一交易日股价推算精确卖出触发价（需结合财报质量与季频排名）",
        blocker_reason=(
            f"仅当季度财报质量指标恶化或 FCF Yield 排位跌出精选区，并于季频再平衡窗口（预计 {next_rebal_str}）确认后才触发卖出。"
        ),
    )

    why_items = [
        {"指标": "模型计算类别", "当前值": "基本面主导（非纯价格）", "说明": "拒绝编造虚假股价触发价"},
        {"指标": "当前精选组合", "当前值": ", ".join(selected) or "空仓", "说明": f"Top {strategy.top_n} 各配 {int(strategy.target_weight_per_stock*100)}%，留 {int(strategy.cash_buffer*100)}% 现金"},
        {"指标": "距下次季频调仓", "当前值": f"约 {days_to_next} 个交易日", "说明": f"预计 {next_rebal_str}（每 {q_days} 交易日）"},
        {"指标": "财报快照覆盖", "当前值": f"{len(prices_df.columns) - len(uncovered)} / {len(prices_df.columns)} 只", "说明": f"未覆盖标的（如 {', '.join(uncovered[:4])}）将被跳过" if uncovered else "全部覆盖"},
    ]
    why_expl = [
        "S08 属于**第三类（不能仅靠股价计算）**策略：买卖决策取决于公司 ROIC、自由现金流、杠杆率与全池估值排名。",
        "特别提示（严谨性边界）：当前回测引擎使用内置基础快照字典 `DEFAULT_FUNDAMENTALS_DATABASE` 配合股价变动近似估算 FCF Yield，实盘决策前必须核对最新 SEC 10-Q/10-K 财报披露。",
        f"系统不会对 S08 编造“明天涨到 $X 卖出 / 跌到 $Y 买入”的伪精确数字，而是提供完整的因子横截面明细与季频调仓日程（预计下一期：{next_rebal_str}）。",
    ]

    return ForwardSignalReport(
        strategy_code="S08",
        strategy_name=f"S08 - {meta['cn_name']}",
        strategy_en_name=meta["en_name"],
        is_portfolio=True,
        ticker_or_pool=f"质量价值因子池 ({len(prices_df.columns)} 只)",
        as_of_date=as_of_str,
        next_trading_date=next_str,
        latest_close=latest_close,
        calculation_category=meta["category"],
        calculation_category_label=meta["category_label"],
        confirmed_signal=confirmed_sig,
        confirmed_is_new_event=False,
        confirmed_position=1 if selected else 0,
        confirmed_summary=confirmed_summary,
        confirmed_execution_timing=confirmed_exec,
        conditional_summary=f"S08 为基本面多因子模型，下一交易日 ({next_str}) 无纯价格触发点；请在季频窗口 ({next_rebal_str}) 结合最新财报数据复核。",
        conditional_execution_timing=f"季频财报复核日 ({next_rebal_str}) 收盘确认 → 次交易日执行",
        current_signal=confirmed_sig,
        next_buy_trigger=buy_trig,
        next_sell_trigger=sell_trig,
        primary_trigger=buy_trig,
        execution_timing=confirmed_exec,
        why_items=why_items,
        why_explanation=why_expl,
        next_rebalance_date=next_rebal_str,
        scenario_table=scenario_df,
        extra={"uncovered_tickers": uncovered},
    )


def _solve_s09(
    prices_df: pd.DataFrame,
    focus_ticker: Optional[str] = None,
) -> ForwardSignalReport:
    meta = STRATEGY_METADATA["S09"]
    strategy = create_portfolio_strategy("S09", ["SPY", "VEU", "BIL"])
    weights_df = strategy.compute_target_weights(prices_df)

    as_of_ts = pd.Timestamp(prices_df.index[-1])
    next_ts = estimate_next_trading_date(as_of_ts)
    today_is_me = is_month_end_business_day(as_of_ts)
    tomorrow_is_me = is_month_end_business_day(next_ts)
    next_rebal_ts = next_month_end_business_day(as_of_ts, include_today=False)

    as_of_str = as_of_ts.strftime("%Y-%m-%d")
    next_str = next_ts.strftime("%Y-%m-%d")
    next_rebal_str = next_rebal_ts.strftime("%Y-%m-%d")

    lb = strategy.lookback_days
    n = len(prices_df)
    us, intl, safe = strategy.us_equity, strategy.intl_equity, strategy.safe_asset
    latest_close = float(prices_df[us].iloc[-1])

    # Current 252-day returns
    if n > lb:
        p_us_now, p_us_base = float(prices_df[us].iloc[-1]), float(prices_df[us].iloc[-1 - lb])
        p_intl_now, p_intl_base = float(prices_df[intl].iloc[-1]), float(prices_df[intl].iloc[-1 - lb])
        p_safe_now, p_safe_base = float(prices_df[safe].iloc[-1]), float(prices_df[safe].iloc[-1 - lb])

        r_us = (p_us_now / p_us_base) - 1.0
        r_intl = (p_intl_now / p_intl_base) - 1.0
        r_safe = (p_safe_now / p_safe_base) - 1.0

        # Tomorrow's base prices (shifted 1 bar forward: iloc[-lb] becomes t+1 - 252!)
        p_us_base_next = float(prices_df[us].iloc[-lb])
        p_intl_base_next = float(prices_df[intl].iloc[-lb])
        p_safe_base_next = float(prices_df[safe].iloc[-lb])
        r_intl_hold = (p_intl_now / p_intl_base_next) - 1.0
        r_safe_hold = (p_safe_now / p_safe_base_next) - 1.0

        # Ceteris paribus SPY critical prices at T+1 (holding VEU & BIL at T prices)
        spy_vs_veu_px = p_us_base_next * (1.0 + r_intl_hold)
        spy_vs_bil_px = p_us_base_next * (1.0 + r_safe_hold)
    else:
        r_us = r_intl = r_safe = np.nan
        p_us_now = float(prices_df[us].iloc[-1])
        p_intl_now = float(prices_df[intl].iloc[-1])
        p_safe_now = float(prices_df[safe].iloc[-1])
        spy_vs_veu_px = spy_vs_bil_px = None

    completed_w = weights_df[weights_df.index < as_of_ts] if not today_is_me else weights_df
    active_w = completed_w.iloc[-1].to_dict() if not completed_w.empty else {us: 0.0, intl: 0.0, safe: 1.0}
    proforma_w = weights_df.iloc[-1].to_dict() if not weights_df.empty else active_w

    active_asset = max(active_w, key=lambda k: active_w[k])
    proforma_asset = max(proforma_w, key=lambda k: proforma_w[k])

    scenario_df = pd.DataFrame(
        [
            {
                "ETF 标的": us,
                "资产角色": "美股标普500 (进攻)",
                "最新价": f"${p_us_now:,.2f}",
                "当前 252 日总回报": f"{r_us*100:+.2f}%" if pd.notna(r_us) else "预热中",
                "当前已确认仓位": f"{active_w.get(us, 0.0)*100:.0f}%",
                "若今日调仓目标": f"{proforma_w.get(us, 0.0)*100:.0f}%",
            },
            {
                "ETF 标的": intl,
                "资产角色": "全球除美股市 (进攻)",
                "最新价": f"${p_intl_now:,.2f}",
                "当前 252 日总回报": f"{r_intl*100:+.2f}%" if pd.notna(r_intl) else "预热中",
                "当前已确认仓位": f"{active_w.get(intl, 0.0)*100:.0f}%",
                "若今日调仓目标": f"{proforma_w.get(intl, 0.0)*100:.0f}%",
            },
            {
                "ETF 标的": safe,
                "资产角色": "短期美债现金 (避险)",
                "最新价": f"${p_safe_now:,.2f}",
                "当前 252 日总回报": f"{r_safe*100:+.2f}%" if pd.notna(r_safe) else "预热中",
                "当前已确认仓位": f"{active_w.get(safe, 0.0)*100:.0f}%",
                "若今日调仓目标": f"{proforma_w.get(safe, 0.0)*100:.0f}%",
            },
        ]
    )

    switched = active_asset != proforma_asset
    if today_is_me and switched:
        confirmed_sig = "REBALANCE"
        confirmed_is_new = True
        confirmed_summary = f"今日 ({as_of_str}) 为月末调仓日，已确认轮动切换：**100% {active_asset} → 100% {proforma_asset}**。"
        confirmed_exec = EXEC_PORTFOLIO_CONFIRMED
    else:
        confirmed_sig = "HOLD"
        confirmed_is_new = False
        confirmed_summary = f"截至 {as_of_str} 维持已确认持仓：**100% {active_asset}**（当前实时 12M 领先资产为 {proforma_asset}）。"
        confirmed_exec = f"非月末调仓日 · 维持 100% {active_asset}，下一调仓检验日为 {next_rebal_str}"

    if tomorrow_is_me:
        switch_thresh_px = max(spy_vs_veu_px or 0.0, spy_vs_bil_px or 0.0) if (spy_vs_veu_px and spy_vs_bil_px) else None
        buy_trig = TriggerCondition(
            action="BUY",
            can_trigger_next_day=True,
            trigger_price=switch_thresh_px if active_asset != us else None,
            comparison=">" if (active_asset != us and switch_thresh_px) else "SCENARIO",
            distance_pct=((switch_thresh_px - p_us_now) / p_us_now) if (active_asset != us and switch_thresh_px) else None,
            distance_abs=(switch_thresh_px - p_us_now) if (active_asset != us and switch_thresh_px) else None,
            condition_summary=(
                f"明日 ({next_str}) 为月末调仓日！若 VEU/BIL 价格不变，SPY 需收于 ≥ ${switch_thresh_px:,.2f} 才可重新夺回 100% 美股仓位；"
                f"当前情景预计切换目标：100% {proforma_asset}。"
                if switch_thresh_px
                else f"明日 ({next_str}) 为月末调仓日，比较 SPY / VEU / BIL 的 252 日收益决定 100% 持仓。"
            ),
            earliest_execution=EXEC_PORTFOLIO_NEXT_DAY_COND,
        )
        sell_trig = TriggerCondition(
            action="SELL",
            can_trigger_next_day=True,
            trigger_price=switch_thresh_px if active_asset == us else None,
            comparison="<" if (active_asset == us and switch_thresh_px) else "SCENARIO",
            distance_pct=((switch_thresh_px - p_us_now) / p_us_now) if (active_asset == us and switch_thresh_px) else None,
            distance_abs=(switch_thresh_px - p_us_now) if (active_asset == us and switch_thresh_px) else None,
            condition_summary=(
                f"明日 ({next_str}) 为月末调仓日！若 VEU/BIL 价格不变，SPY 跌破 ${switch_thresh_px:,.2f} 将触发切出美股"
                f"（跌破 ${spy_vs_veu_px:,.2f} 输给 VEU，跌破 ${spy_vs_bil_px:,.2f} 跑输 BIL 转避险）。"
                if (switch_thresh_px and spy_vs_veu_px and spy_vs_bil_px)
                else f"明日 ({next_str}) 为月末调仓日，若当前持仓 {active_asset} 失去 12M 动量优势将触发全仓切换。"
            ),
            earliest_execution=EXEC_PORTFOLIO_NEXT_DAY_COND,
        )
        cond_summary = f"明日 ({next_str}) 即为月末双动量调仓日！按最新 252 日回报，预计目标仓位为 100% {proforma_asset}。"
        cond_exec = EXEC_PORTFOLIO_NEXT_DAY_COND
    else:
        buy_trig = TriggerCondition(
            action="BUY",
            can_trigger_next_day=False,
            comparison="BLOCKED",
            condition_summary=f"明日 ({next_str}) 非月末调仓日，不会触发轮动买入",
            blocker_reason=f"S09 严格按月末最后一个交易日检验，下一检验日为 {next_rebal_str}。",
        )
        sell_trig = TriggerCondition(
            action="SELL",
            can_trigger_next_day=False,
            comparison="BLOCKED",
            condition_summary=f"明日 ({next_str}) 非月末调仓日，不会触发轮动卖出",
            blocker_reason=f"须等待至 {next_rebal_str} 收盘比较 SPY / VEU / BIL 的 252 日回报；若当前持仓 {active_asset} 落败则切换。",
        )
        cond_summary = f"明日 ({next_str}) 非月末调仓日，继续持有 100% {active_asset}；下一检验日为 {next_rebal_str}。"
        cond_exec = f"最早检验日：{next_rebal_str} 收盘确认 → 次月首个交易日执行"

    why_items = [
        {"指标": "当前已确认持仓", "当前值": f"100% {active_asset}", "说明": "单资产全仓轮动"},
        {"指标": "当前 12M 领先资产", "当前值": f"{proforma_asset}", "说明": f"SPY {r_us*100:+.1f}% vs VEU {r_intl*100:+.1f}% vs BIL {r_safe*100:+.1f}%" if pd.notna(r_us) else "预热中"},
        {"指标": "SPY 相对 VEU 临界参考*", "当前值": f"${spy_vs_veu_px:,.2f}" if spy_vs_veu_px else "N/A", "说明": "假设 VEU 不变时的 SPY 换仓线"},
        {"指标": "SPY 相对 BIL 避险线*", "当前值": f"${spy_vs_bil_px:,.2f}" if spy_vs_bil_px else "N/A", "说明": "假设 BIL 不变时的绝对动量防守线"},
        {"指标": "下一月末检验日", "当前值": next_rebal_str, "说明": "月末确认 → 次月首交易日执行"},
    ]
    why_expl = [
        "S09 双重动量同时依赖 `SPY`、`VEU`、`BIL` 三只 ETF 的 252 日总回报及月末调仓日历。",
        "第一步（相对动量）：比较 `R_252(SPY)` 与 `R_252(VEU)` 选出进攻胜者；第二步（绝对动量）：胜者必须跑赢短期美债 `R_252(BIL)` 才持有，否则 100% 切入 `BIL` 避险。",
        f"上表给出“其余 ETF 价格不变”情景下的 SPY 切换参考线，便于盘中直观跟踪美股与国际股市/国债的动量距离。",
    ]

    return ForwardSignalReport(
        strategy_code="S09",
        strategy_name=f"S09 - {meta['cn_name']}",
        strategy_en_name=meta["en_name"],
        is_portfolio=True,
        ticker_or_pool="宏观双动量 (SPY / VEU / BIL)",
        as_of_date=as_of_str,
        next_trading_date=next_str,
        latest_close=latest_close,
        calculation_category=meta["category"],
        calculation_category_label=meta["category_label"],
        confirmed_signal=confirmed_sig,
        confirmed_is_new_event=confirmed_is_new,
        confirmed_position=1,
        confirmed_summary=confirmed_summary,
        confirmed_execution_timing=confirmed_exec,
        conditional_summary=cond_summary,
        conditional_execution_timing=cond_exec,
        current_signal=confirmed_sig,
        next_buy_trigger=buy_trig,
        next_sell_trigger=sell_trig,
        primary_trigger=sell_trig if (tomorrow_is_me and active_asset == us) else buy_trig,
        execution_timing=confirmed_exec if confirmed_is_new else cond_exec,
        why_items=why_items,
        why_explanation=why_expl,
        next_rebalance_date=next_rebal_str,
        scenario_table=scenario_df,
    )
