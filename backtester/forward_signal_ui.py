"""
Streamlit UI Renderer for the Next-Day Signal Preview Dashboard.

Follows Streamlit 1.67+ best practices:
- Native Streamlit containers, metrics, badges, progress bars, segmented controls, and ECharts.
- No deprecated `use_container_width`; uses `width="stretch"` and accessible `alt=` text.
- Expensive Historical Replay Validation runs only on explicit user trigger.
"""

from typing import Any, Callable, Dict, List, Optional
import pandas as pd
import streamlit as st

from .forward_signal import (
    ForwardSignalReport,
    generate_default_price_grid,
    parse_custom_prices,
    run_cross_strategy_sensitivity,
    run_historical_replay_validation,
    run_portfolio_replay_validation,
    run_price_sensitivity,
)


_SIGNAL_BADGE_MAP = {
    "BUY": ("green", ":material/trending_up: BUY · 买入"),
    "SELL": ("red", ":material/trending_down: SELL · 卖出"),
    "REBALANCE": ("orange", ":material/balance: REBALANCE · 调仓"),
    "HOLD": ("blue", ":material/pause_circle: HOLD · 持仓观望"),
    "WAIT": ("gray", ":material/hourglass_empty: WAIT · 空仓观望"),
}

_CATEGORY_BADGE_MAP = {
    "EXACT_THRESHOLD": "violet",
    "DYNAMIC_SCENARIO": "blue",
    "FUNDAMENTAL_BOUNDED": "orange",
}

_VALIDATOR_LABEL_MAP = {
    "EXECUTABLE": ("green", ":material/verified: 校验通过"),
    "CAUTION": ("orange", ":material/warning: 需关注风控"),
    "BLOCKED": ("red", ":material/block: 执行受限"),
}


def _build_sensitivity_echarts(
    strategy_code: str,
    sens_rows: List[Any],
    latest_close: float,
    ticker: str,
) -> Optional[Dict[str, Any]]:
    """Builds an interactive ECharts chart illustrating dynamic indicator curves across test prices."""
    if not sens_rows:
        return None

    x_labels = [f"${r.test_price:,.2f}" for r in sens_rows]

    if strategy_code == "S02":
        rsi_vals = [
            round(float(r.indicator_snapshot["RSI"]), 2)
            if r.indicator_snapshot.get("RSI") is not None
            else None
            for r in sens_rows
        ]
        return {
            "backgroundColor": "transparent",
            "tooltip": {"trigger": "axis"},
            "legend": {"top": 4, "textStyle": {"color": "#CBD5E1"}},
            "grid": {"left": "8%", "right": "5%", "top": 42, "bottom": 36},
            "xAxis": {
                "type": "category",
                "data": x_labels,
                "name": "假设明日收盘价",
                "nameLocation": "middle",
                "nameGap": 24,
                "axisLabel": {"color": "#94A3B8"},
            },
            "yAxis": {
                "type": "value",
                "min": 0,
                "max": 100,
                "name": "RSI(14)",
                "axisLabel": {"color": "#94A3B8"},
                "splitLine": {"lineStyle": {"color": "rgba(148,163,184,0.15)"}},
            },
            "series": [
                {
                    "name": f"{ticker} 明日动态 RSI(14)",
                    "type": "line",
                    "smooth": True,
                    "data": rsi_vals,
                    "lineStyle": {"width": 3, "color": "#38BDF8"},
                    "itemStyle": {"color": "#38BDF8"},
                    "markLine": {
                        "silent": True,
                        "data": [
                            {"yAxis": 30, "name": "超卖 30", "lineStyle": {"color": "#22C55E", "type": "dashed"}},
                            {"yAxis": 70, "name": "超买 70", "lineStyle": {"color": "#EF4444", "type": "dashed"}},
                        ],
                    },
                }
            ],
        }

    if strategy_code == "S05":
        close_vals = [round(r.test_price, 2) for r in sens_rows]
        up_vals = [
            round(float(r.indicator_snapshot["BB_Upper"]), 2)
            if r.indicator_snapshot.get("BB_Upper") is not None
            else None
            for r in sens_rows
        ]
        mid_vals = [
            round(float(r.indicator_snapshot["BB_Middle"]), 2)
            if r.indicator_snapshot.get("BB_Middle") is not None
            else None
            for r in sens_rows
        ]
        return {
            "backgroundColor": "transparent",
            "tooltip": {"trigger": "axis"},
            "legend": {"top": 4, "textStyle": {"color": "#CBD5E1"}},
            "grid": {"left": "9%", "right": "5%", "top": 42, "bottom": 36},
            "xAxis": {
                "type": "category",
                "data": x_labels,
                "axisLabel": {"color": "#94A3B8"},
            },
            "yAxis": {
                "type": "value",
                "scale": True,
                "axisLabel": {"color": "#94A3B8", "formatter": "${value}"},
                "splitLine": {"lineStyle": {"color": "rgba(148,163,184,0.15)"}},
            },
            "series": [
                {"name": "假设收盘价", "type": "line", "data": close_vals, "lineStyle": {"width": 2.5, "color": "#F8FAFC"}},
                {"name": "动态布林上轨 (2σ)", "type": "line", "data": up_vals, "lineStyle": {"width": 2, "color": "#22C55E", "type": "dashed"}},
                {"name": "动态布林中轨 (SMA20)", "type": "line", "data": mid_vals, "lineStyle": {"width": 2, "color": "#F59E0B", "type": "dashed"}},
            ],
        }

    if strategy_code == "S01":
        ma_s = [
            round(float(r.indicator_snapshot["MA_Short"]), 2)
            if r.indicator_snapshot.get("MA_Short") is not None
            else None
            for r in sens_rows
        ]
        ma_l = [
            round(float(r.indicator_snapshot["MA_Long"]), 2)
            if r.indicator_snapshot.get("MA_Long") is not None
            else None
            for r in sens_rows
        ]
        return {
            "backgroundColor": "transparent",
            "tooltip": {"trigger": "axis"},
            "legend": {"top": 4, "textStyle": {"color": "#CBD5E1"}},
            "grid": {"left": "9%", "right": "5%", "top": 42, "bottom": 36},
            "xAxis": {"type": "category", "data": x_labels, "axisLabel": {"color": "#94A3B8"}},
            "yAxis": {
                "type": "value",
                "scale": True,
                "axisLabel": {"color": "#94A3B8", "formatter": "${value}"},
                "splitLine": {"lineStyle": {"color": "rgba(148,163,184,0.15)"}},
            },
            "series": [
                {"name": "明日 MA_Short (20)", "type": "line", "data": ma_s, "lineStyle": {"width": 2.5, "color": "#38BDF8"}},
                {"name": "明日 MA_Long (50)", "type": "line", "data": ma_l, "lineStyle": {"width": 2.5, "color": "#F97316"}},
            ],
        }

    return None


def render_compact_signal_banner(
    report: ForwardSignalReport,
    on_open_full: Optional[Callable[[], None]] = None,
) -> None:
    """Renders a compact, wrap-safe Next-Day Signal Preview card inside the Historical Backtest tab."""
    badge_color, badge_text = _SIGNAL_BADGE_MAP.get(report.current_signal, ("gray", report.current_signal))
    cat_color = _CATEGORY_BADGE_MAP.get(report.calculation_category, "gray")
    rv = report.risk_validation
    val_color, val_text = _VALIDATOR_LABEL_MAP.get(rv.status, ("gray", "已校验")) if rv else ("gray", "已校验")

    with st.container(border=True):
        c1, c2 = st.columns([3.6, 1.4], vertical_alignment="center")
        with c1:
            st.markdown(f"#### :material/insights: 下一交易日信号预览速览 · {report.ticker_or_pool} ({report.strategy_code})")
            st.caption(
                f"截至最新收盘日 {report.as_of_date} (收盘 ${report.latest_close:,.2f}) → 监测下一交易日 {report.next_trading_date} 触发条件"
            )
        with c2:
            if on_open_full is not None:
                st.button(
                    "进入完整信号预览 →",
                    on_click=on_open_full,
                    type="primary",
                    width="stretch",
                    key="jump_to_forward_preview_btn",
                )

        with st.container(horizontal=True):
            st.badge(badge_text, color=badge_color)
            st.badge(report.calculation_category_label, color=cat_color)
            st.badge(val_text, color=val_color)

        b1, b2 = st.columns(2, border=True)
        with b1:
            st.caption("下一日买入触发条件 (Next Buy Trigger)")
            st.markdown(f"**{report.next_buy_trigger.format_trigger_label()}**")
            if report.next_buy_trigger.can_trigger_next_day and report.next_buy_trigger.distance_pct is not None:
                st.caption(f"距当前收盘价：{report.next_buy_trigger.distance_pct*100:+.2f}% (${report.next_buy_trigger.distance_abs:+,.2f})")
            else:
                st.caption(report.next_buy_trigger.blocker_reason or report.next_buy_trigger.condition_summary)
        with b2:
            st.caption("下一日卖出触发条件 (Next Sell Trigger)")
            st.markdown(f"**{report.next_sell_trigger.format_trigger_label()}**")
            if report.next_sell_trigger.can_trigger_next_day and report.next_sell_trigger.distance_pct is not None:
                st.caption(f"距当前收盘价：{report.next_sell_trigger.distance_pct*100:+.2f}% (${report.next_sell_trigger.distance_abs:+,.2f})")
            else:
                st.caption(report.next_sell_trigger.blocker_reason or report.next_sell_trigger.condition_summary)

        st.caption(f"⏱️ {report.execution_timing}")


def render_next_day_signal_dashboard(
    report: ForwardSignalReport,
    single_df: Optional[pd.DataFrame] = None,
    prices_df: Optional[pd.DataFrame] = None,
    ticker: str = "NVDA",
    universe: Optional[List[str]] = None,
) -> None:
    """
    Renders the complete Next-Day Signal Preview (Forward Signal Analysis) dashboard.
    Uses wrap-safe 2-column hero cards + status badges so long price ranges and Chinese labels never truncate.
    """
    badge_color, badge_text = _SIGNAL_BADGE_MAP.get(report.current_signal, ("gray", report.current_signal))
    cat_color = _CATEGORY_BADGE_MAP.get(report.calculation_category, "gray")
    rv = report.risk_validation
    val_color, val_text = _VALIDATOR_LABEL_MAP.get(rv.status, ("gray", "已校验")) if rv else ("gray", "已校验")
    primary = report.primary_trigger

    st.subheader(":material/insights: Next-Day Signal Preview（下一交易日信号预览）")
    st.caption(
        "核心边界声明：本模块严格调用与历史回测完全一致的策略逻辑，精确计算「下一交易日必须满足什么条件才会触发新信号」，而非预测明日必然涨跌。"
    )

    # ==========================================================
    # 1. HERO SIGNAL CARD (2-Column Wide Cards, Zero Truncation)
    # ==========================================================
    with st.container(border=True):
        h_left, h_right = st.columns([2.8, 2.2], vertical_alignment="center")
        with h_left:
            st.markdown(f"### {report.ticker_or_pool} · {report.strategy_en_name}")
            st.caption(
                f"策略：{report.strategy_name} ｜ 截至最新完成交易日：**{report.as_of_date}** ｜ 下一交易日：**{report.next_trading_date}**"
                + (f" ｜ 下一调仓检验日：**{report.next_rebalance_date}**" if report.next_rebalance_date else "")
            )
        with h_right:
            with st.container(horizontal=True):
                st.badge(badge_text, color=badge_color)
                st.badge(report.calculation_category_label, color=cat_color)
                st.badge(val_text, color=val_color)

        st.divider()

        # Primary 2-Column Wide Price & Trigger Cards (wraps cleanly, zero ellipsis)
        k_left, k_right = st.columns(2, border=True)
        with k_left:
            st.caption(f"最新收盘价 · 截至 {report.as_of_date}")
            st.markdown(f"## ${report.latest_close:,.2f}")
            if rv and rv.risk_level:
                score_txt = f"（风险评分 {rv.risk_score:.1f}/100）" if rv.risk_score is not None else ""
                st.caption(f"🛡️ 风控评级：**{rv.risk_level}**{score_txt}")
            else:
                st.caption(f"📌 计算类别：{report.calculation_category_label}")

        with k_right:
            action_cn = (
                "下一日潜在买入触发条件 (Next Buy Trigger)"
                if (primary and primary.action == "BUY")
                else ("下一日潜在卖出触发条件 (Next Sell Trigger)" if (primary and primary.action == "SELL") else "下一日潜在触发条件")
            )
            st.caption(action_cn)
            trigger_label = primary.format_trigger_label() if primary else "当前无法触发"
            st.markdown(f"## {trigger_label}")

            if primary and primary.can_trigger_next_day and primary.distance_pct is not None and primary.distance_abs is not None:
                dist_color = "green" if primary.distance_pct >= 0 else "red"
                dist_icon = ":material/arrow_upward:" if primary.distance_pct >= 0 else ":material/arrow_downward:"
                with st.container(horizontal=True, vertical_alignment="center"):
                    st.badge(
                        f"{dist_icon} 距当前价 {primary.distance_pct*100:+.2f}% (${primary.distance_abs:+,.2f})",
                        color=dist_color,
                    )
                    with st.popover("查看完整规则"):
                        st.markdown(primary.condition_summary)
                        st.caption(f"⏱️ {primary.earliest_execution or report.conditional_execution_timing}")
            elif primary:
                detail_msg = primary.blocker_reason or primary.condition_summary
                st.caption(detail_msg)

        # Visual Distance Progress Bar (Mockup 1)
        if primary and primary.can_trigger_next_day and primary.distance_pct is not None:
            abs_dist = abs(primary.distance_pct)
            # Proximity meter: 100% when 0% away, smoothly scales down across 0%~25% distance
            proximity = max(0.04, min(1.0, 1.0 - min(abs_dist, 0.25) / 0.25))
            p_left, p_right = st.columns([3, 1])
            with p_left:
                st.markdown("**距离触发价接近度 (Distance to Trigger)**")
            with p_right:
                st.markdown(f"<div style='text-align:right; font-weight:700;'>{primary.distance_pct*100:+.2f}%</div>", unsafe_allow_html=True)
            st.progress(proximity)

        # Two Signal Tiers (Mockup 2: Confirmed Signal vs Conditional Signal)
        tier1, tier2 = st.columns(2, border=True)
        with tier1:
            st.markdown("#### :material/check_circle: Confirmed Signal（已确认信号）")
            st.markdown(report.confirmed_summary)
            st.caption(f"⏱️ **{report.confirmed_execution_timing}**")
        with tier2:
            st.markdown("#### :material/track_changes: Conditional Signal（条件性信号）")
            st.markdown(report.conditional_summary)
            st.caption(f"⏱️ **{report.conditional_execution_timing}**")

        # 5-Field Core Specification Summary
        five_rows = [
            {
                "核心项": "Current Signal (当前信号)",
                "结果": f"{report.current_signal}（{'新确认拐点' if report.confirmed_is_new_event else '维持当前状态'}）",
                "说明": report.confirmed_summary,
            },
            {
                "核心项": "Next Buy Trigger (下一日买入触发)",
                "结果": report.next_buy_trigger.format_trigger_label(),
                "说明": report.next_buy_trigger.condition_summary
                if report.next_buy_trigger.can_trigger_next_day
                else (report.next_buy_trigger.blocker_reason or report.next_buy_trigger.condition_summary),
            },
            {
                "核心项": "Next Sell Trigger (下一日卖出触发)",
                "结果": report.next_sell_trigger.format_trigger_label(),
                "说明": report.next_sell_trigger.condition_summary
                if report.next_sell_trigger.can_trigger_next_day
                else (report.next_sell_trigger.blocker_reason or report.next_sell_trigger.condition_summary),
            },
            {
                "核心项": "Distance to Trigger (距离触发条件)",
                "结果": (
                    f"{primary.distance_pct*100:+.2f}% (${primary.distance_abs:+,.2f})"
                    if (primary and primary.can_trigger_next_day and primary.distance_pct is not None and primary.distance_abs is not None)
                    else ("多资产/日历条件决定" if report.is_portfolio else "当前无有效单日触发价")
                ),
                "说明": "相对最新完成交易日收盘价的涨跌幅与价差",
            },
            {
                "核心项": "Execution Timing (最早执行时点)",
                "结果": "下一交易日 (T+1)" if report.confirmed_is_new_event else (
                    f"若明日触发则最早 T+2" if (primary and primary.can_trigger_next_day) else (
                        f"下一检验日 ({report.next_rebalance_date}) 次日" if report.next_rebalance_date else "暂无可执行触发"
                    )
                ),
                "说明": report.execution_timing,
            },
        ]
        st.markdown("##### :material/fact_check: 五项核心决策信息总览")
        st.dataframe(
            pd.DataFrame(five_rows),
            width="stretch",
            hide_index=True,
            alt="Next-day signal five core decision fields summary",
        )

    # ==========================================================
    # 2. WHY THIS SIGNAL? & RISK VALIDATOR (Side by side or stacked)
    # ==========================================================
    col_why, col_risk = st.columns([1.35, 1.0])
    with col_why:
        with st.container(border=True):
            st.markdown("#### :material/psychology: Why This Signal?（信号计算依据）")
            if report.why_items:
                st.dataframe(
                    pd.DataFrame(report.why_items),
                    width="stretch",
                    hide_index=True,
                    alt="Indicators and thresholds used in signal calculation",
                )
            for bullet in report.why_explanation:
                st.markdown(f"- {bullet}")

    with col_risk:
        with st.container(border=True):
            st.markdown("#### :material/verified_user: Risk & Execution Validator（风控与执行校验）")
            rv = report.risk_validation
            if rv is not None:
                st.markdown(f"- **持仓状态**：{rv.current_position_state}")
                st.markdown(f"- **资金/现金校验**：{rv.cash_check}")
                st.markdown(f"- **仓位与风险上限**：{rv.position_limit_check}")
                st.markdown(f"- **数据与事件护栏**：{rv.risk_guardrail_check}")
                st.markdown(f"- **防未来函数执行时点**：{rv.earliest_execution_rule}")
                for w in rv.warnings:
                    st.warning(w, title="实盘风控提示")

    # Portfolio Multi-Asset Scenario Table (S04 / S07 / S08 / S09)
    if report.scenario_table is not None and not report.scenario_table.empty:
        with st.container(border=True):
            st.markdown(f"#### :material/table_view: 多资产情景与排名 / 容忍带明细 ({report.strategy_code})")
            st.dataframe(
                report.scenario_table,
                width="stretch",
                hide_index=True,
                alt=f"Multi-asset scenario and ranking details for {report.strategy_code}",
            )

    # ==========================================================
    # 3. EXTRA HIGH-VALUE TOOLS: PRICE SENSITIVITY & REPLAY VALIDATION
    # ==========================================================
    st.markdown("---")
    tool_choice = st.segmented_control(
        "深度验证与推演工具",
        options=[
            "🎚️ 价格敏感性分析 (Price Sensitivity)",
            "🧪 历史重放验证 (Historical Replay Validation)",
        ],
        default="🎚️ 价格敏感性分析 (Price Sensitivity)",
        key="forward_tool_mode",
    )

    if tool_choice == "🎚️ 价格敏感性分析 (Price Sensitivity)":
        _render_price_sensitivity_section(report, single_df=single_df, ticker=ticker)
    elif tool_choice == "🧪 历史重放验证 (Historical Replay Validation)":
        _render_replay_validation_section(
            report,
            single_df=single_df,
            prices_df=prices_df,
            ticker=ticker,
            universe=universe,
        )


def _render_price_sensitivity_section(
    report: ForwardSignalReport,
    single_df: Optional[pd.DataFrame],
    ticker: str,
) -> None:
    with st.container(border=True):
        st.markdown("#### :material/tune: 价格敏感性分析 (Price Sensitivity Analysis)")
        st.caption(
            "输入任意假设的下一交易日收盘价（例如 `160, 170, 180, 190`），系统将直接代入真实策略引擎计算各价位下的 BUY / SELL / HOLD / WAIT 响应及动态指标值。"
        )

        if single_df is None or single_df.empty:
            st.info(
                "当前为多资产组合策略，其信号由多只资产的相对表现与调仓日历共同决定；如需单股价格敏感性推演，请在上方选择焦点成分股或切换至单股策略。",
                title="组合策略提示",
            )
            return

        base_px = float(single_df["Close"].iloc[-1])
        extras: List[Optional[float]] = []
        if not report.is_portfolio and report.primary_trigger:
            extras.append(report.primary_trigger.trigger_price)
            if report.primary_trigger.price_range:
                extras.extend(list(report.primary_trigger.price_range))

        default_grid = generate_default_price_grid(base_px, extra_prices=extras)
        default_str = ", ".join(f"{p:.2f}" for p in default_grid)

        sens_ctx = (report.strategy_code, ticker, round(base_px, 2))
        if st.session_state.get("_sens_ctx") != sens_ctx:
            st.session_state["_sens_ctx"] = sens_ctx
            st.session_state["sens_prices_input"] = default_str

        in_col, mode_col = st.columns([2.6, 1.4], vertical_alignment="bottom")
        with in_col:
            raw_prices = st.text_input(
                f"自定义假设明日收盘价列表 ($ USD，逗号分隔 · {ticker} 最新价 ${base_px:,.2f})",
                key="sens_prices_input",
            )
        with mode_col:
            show_cross = st.toggle(
                "同时对比 S01/S02/S03/S05/S06 全策略矩阵",
                value=True,
                key="sens_cross_toggle",
            )

        try:
            test_prices = parse_custom_prices(raw_prices)
            if not test_prices:
                st.warning("请输入至少一个大于 0 的有效价格数字。")
                return
        except ValueError:
            st.error("价格格式无法解析，请使用英文或中文逗号分隔的数字，例如：`160, 170, 180, 190`。")
            return

        if not report.is_portfolio:
            sens_rows, sens_df = run_price_sensitivity(report.strategy_code, single_df, test_prices)
            st.markdown(f"**当前策略 ({report.strategy_name}) 在各假设收盘价下的实时响应与动态指标：**")
            st.dataframe(
                sens_df,
                width="stretch",
                hide_index=True,
                alt=f"Price sensitivity table for {report.strategy_code} on {ticker}",
            )

            chart_opt = _build_sensitivity_echarts(report.strategy_code, sens_rows, base_px, ticker)
            if chart_opt is not None:
                st.echarts_chart(
                    chart_opt,
                    width="stretch",
                    height=300,
                    alt=f"Dynamic indicator sensitivity chart for {report.strategy_code} on {ticker}",
                    key=f"sens_chart_{report.strategy_code}_{ticker}",
                )

        if show_cross:
            st.markdown(f"**跨策略横向对比矩阵（{ticker} · 相同假设明日收盘价下 5 大单股策略信号对比）：**")
            cross_df = run_cross_strategy_sensitivity(single_df, test_prices)
            st.dataframe(
                cross_df,
                width="stretch",
                hide_index=True,
                alt=f"Cross-strategy signal matrix for {ticker} across hypothetical prices",
            )


def _render_replay_validation_section(
    report: ForwardSignalReport,
    single_df: Optional[pd.DataFrame],
    prices_df: Optional[pd.DataFrame],
    ticker: str,
    universe: Optional[List[str]],
) -> None:
    with st.container(border=True):
        st.markdown("#### :material/history: 历史重放验证 (Historical Replay Validation)")
        st.caption(
            "逐日把历史某天当作「今天」，严格隐藏之后所有未来行情，用预览模块计算下一日触发条件并代入真实次日行情，核验与原始回测引擎的一致率（防未来函数与逻辑漂移的最高标准审计）。"
        )

        replay_state_key = f"replay_result_{report.strategy_code}_{report.ticker_or_pool}"

        c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
        if not report.is_portfolio:
            with c1:
                replay_days = st.segmented_control(
                    "重放检验交易日数",
                    options=[20, 60, 120],
                    default=60,
                    format_func=lambda d: f"最近 {d} 个交易日",
                    key="replay_days_selector",
                ) or 60
            with c2:
                run_replay = st.button(
                    "▶️ 开始无前视历史重放验证",
                    type="primary",
                    width="stretch",
                    key="run_replay_btn",
                )
        else:
            with c1:
                replay_months = st.segmented_control(
                    "重放检验调仓节点数",
                    options=[6, 12, 24],
                    default=12,
                    format_func=lambda m: f"最近 {m} 个调仓节点",
                    key="replay_months_selector",
                ) or 12
            with c2:
                run_replay = st.button(
                    "▶️ 开始无前视调仓重放验证",
                    type="primary",
                    width="stretch",
                    key="run_replay_btn",
                )

        if run_replay:
            with st.spinner("⏳ 正在执行严格无前视历史逐日重放校验，请稍候..."):
                if not report.is_portfolio and single_df is not None:
                    res = run_historical_replay_validation(
                        strategy_code=report.strategy_code,
                        df=single_df,
                        ticker=ticker,
                        replay_days=int(replay_days),
                    )
                elif prices_df is not None:
                    res = run_portfolio_replay_validation(
                        strategy_code=report.strategy_code,
                        prices_df=prices_df,
                        universe=list(universe) if universe else list(prices_df.columns),
                        replay_months=int(replay_months),
                    )
                else:
                    st.warning("缺少可用历史数据进行重放验证。")
                    return
                st.session_state[replay_state_key] = res

        rv = st.session_state.get(replay_state_key)
        if rv is not None:
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.metric("校验样本数", f"{rv.total_days_tested} 次", border=True)
            with r2:
                st.metric("情景引擎一致率", f"{rv.scenario_accuracy_pct:.2f}%", border=True)
            with r3:
                st.metric("阈值反推一致率", f"{rv.threshold_accuracy_pct:.2f}%", border=True)
            with r4:
                st.metric(
                    "买卖拐点命中",
                    f"买 {rv.buy_transitions_matched}/{rv.buy_transitions_actual} · 卖 {rv.sell_transitions_matched}/{rv.sell_transitions_actual}",
                    border=True,
                )

            if rv.mismatches == 0:
                st.success(
                    f"验证通过：在 {rv.total_days_tested} 个历史截断样本上，下一日信号预览模块与原始回测引擎保持 100.00% 完全一致（0 偏差）。",
                    title="零偏差审计通过",
                )
            else:
                st.warning(
                    f"检测到 {rv.mismatches} 条存在边界差异的样本，请展开下方逐日审计表核查。",
                    title="存在边界差异",
                )

            for note in rv.notes:
                st.markdown(f"- {note}")

            if not rv.details_df.empty:
                st.dataframe(
                    rv.details_df,
                    width="stretch",
                    hide_index=True,
                    height=340,
                    alt=f"Historical replay validation audit table for {report.strategy_code}",
                )
