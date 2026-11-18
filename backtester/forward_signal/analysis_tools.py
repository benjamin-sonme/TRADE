"""
Additional High-Value Verification & Decision-Support Modules:
1. Historical Replay Validation (历史重放验证 - strict no-lookahead audit)
2. Price Sensitivity Analysis (价格敏感性分析 - single strategy & cross-strategy matrix)
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from .models import ReplayValidationResult, SensitivityRow, TriggerCondition
from .scenario_engine import (
    ForwardScenarioEngine,
    classify_transition,
    create_portfolio_strategy,
    create_single_stock_strategy,
    is_month_end_business_day,
)
from .threshold_solver import solve_single_stock_forward_signal


# ---------------------------------------------------------------------------
# 1. Price Sensitivity Analysis
# ---------------------------------------------------------------------------

def generate_default_price_grid(
    latest_close: float,
    extra_prices: Optional[Sequence[Optional[float]]] = None,
) -> List[float]:
    """Generates a balanced sensitivity price grid around `latest_close` plus any solved trigger prices."""
    multipliers = [0.85, 0.90, 0.95, 0.98, 1.00, 1.02, 1.05, 1.10, 1.15]
    prices = [round(latest_close * m, 2) for m in multipliers if latest_close * m > 0.01]
    if extra_prices:
        for p in extra_prices:
            if p is not None and np.isfinite(p) and p > 0.01:
                prices.append(round(float(p), 2))
    return sorted(set(prices))


def parse_custom_prices(raw_text: str) -> List[float]:
    """Parses comma/space separated user price inputs like '160, 170, 180, 190'."""
    if not raw_text:
        return []
    cleaned = raw_text.replace("，", ",").replace(";", ",").replace(" ", ",")
    out: List[float] = []
    for token in cleaned.split(","):
        tok = token.strip().lstrip("$")
        if not tok:
            continue
        val = float(tok)
        if val > 0:
            out.append(round(val, 4))
    return sorted(set(out))


def run_price_sensitivity(
    strategy_code: str,
    df: pd.DataFrame,
    test_prices: Sequence[float],
    strategy=None,
) -> Tuple[List[SensitivityRow], pd.DataFrame]:
    """
    Evaluates a single-stock strategy across a list of hypothetical next-day closing prices
    using the EXACT `strategy.generate_signals(sim_df)` engine.
    Returns `(List[SensitivityRow], pd.DataFrame)`.
    """
    code = strategy_code.upper()
    if strategy is None:
        strategy = create_single_stock_strategy(code)

    latest_close = float(df["Close"].iloc[-1])
    rows: List[SensitivityRow] = []
    records: List[Dict[str, Any]] = []

    for px in sorted(set(float(p) for p in test_prices if p > 0)):
        new_pos, prev_pos, action, snap = ForwardScenarioEngine.evaluate_next_day_price(df, strategy, px)
        pct_chg = (px - latest_close) / latest_close if latest_close > 0 else 0.0
        rows.append(
            SensitivityRow(
                test_price=px,
                pct_change=pct_chg,
                resulting_signal=action,
                raw_position=new_pos,
                is_transition=(new_pos != prev_pos),
                indicator_snapshot=snap,
            )
        )

        rec: Dict[str, Any] = {
            "假设明日收盘价": f"${px:,.2f}",
            "较最新价涨跌幅": f"{pct_chg*100:+.2f}%",
            "策略信号": action,
            "目标仓位": "持多 (1)" if new_pos == 1 else "空仓 (0)",
        }
        # Attach strategy-specific indicator snapshots directly from real output
        if code == "S01":
            rec["短期均线"] = f"${snap['MA_Short']:,.2f}" if snap.get("MA_Short") is not None else "N/A"
            rec["长期均线"] = f"${snap['MA_Long']:,.2f}" if snap.get("MA_Long") is not None else "N/A"
        elif code == "S02":
            rec["明日 RSI(14)"] = f"{snap['RSI']:.2f}" if snap.get("RSI") is not None else "N/A"
            rec["明日 SMA(200)"] = f"${snap['Trend_MA']:,.2f}" if snap.get("Trend_MA") is not None else "预热中"
        elif code == "S03":
            rec["20日上轨"] = f"${snap['Upper_Channel']:,.2f}" if snap.get("Upper_Channel") is not None else "N/A"
            rec["10日下轨"] = f"${snap['Lower_Channel']:,.2f}" if snap.get("Lower_Channel") is not None else "N/A"
        elif code == "S05":
            rec["动态布林上轨"] = f"${snap['BB_Upper']:,.2f}" if snap.get("BB_Upper") is not None else "N/A"
            rec["动态布林中轨"] = f"${snap['BB_Middle']:,.2f}" if snap.get("BB_Middle") is not None else "N/A"
            rec["近5日挤压"] = "是" if snap.get("Recent_Squeeze") else "否"
        elif code == "S06":
            rec["SMA20"] = f"${snap['SMA_Fast']:,.2f}" if snap.get("SMA_Fast") is not None else "N/A"
            rec["SMA50"] = f"${snap['SMA_Mid']:,.2f}" if snap.get("SMA_Mid") is not None else "N/A"
            rec["SMA200"] = f"${snap['SMA_Slow']:,.2f}" if snap.get("SMA_Slow") is not None else "N/A"

        records.append(rec)

    return rows, pd.DataFrame(records)


def run_cross_strategy_sensitivity(
    df: pd.DataFrame,
    test_prices: Sequence[float],
    strategy_codes: Sequence[str] = ("S01", "S02", "S03", "S05", "S06"),
) -> pd.DataFrame:
    """
    Compares BUY/SELL/HOLD/WAIT responses of ALL single-stock strategies simultaneously
    across the same hypothetical next-day closing prices (e.g. $160, $170, $180, $190).
    """
    latest_close = float(df["Close"].iloc[-1])
    strategies = {code: create_single_stock_strategy(code) for code in strategy_codes}
    col_labels = {
        "S01": "S01 双均线",
        "S02": "S02 RSI回归",
        "S03": "S03 唐奇安",
        "S05": "S05 布林挤压",
        "S06": "S06 趋势回撤",
    }

    records = []
    for px in sorted(set(float(p) for p in test_prices if p > 0)):
        pct = (px - latest_close) / latest_close if latest_close > 0 else 0.0
        row: Dict[str, Any] = {
            "假设明日收盘价": f"${px:,.2f}",
            "涨跌幅": f"{pct*100:+.2f}%",
        }
        for code in strategy_codes:
            _, _, action, _ = ForwardScenarioEngine.evaluate_next_day_price(df, strategies[code], px)
            badge = {
                "BUY": "🟢 BUY",
                "SELL": "🔴 SELL",
                "HOLD": "🟦 HOLD",
                "WAIT": "⚪ WAIT",
            }.get(action, action)
            row[col_labels.get(code, code)] = badge
        records.append(row)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 2. Historical Replay Validation (Strict No-Lookahead Audit)
# ---------------------------------------------------------------------------

def _predict_from_trigger(
    curr_pos: int,
    trigger: TriggerCondition,
    actual_next_close: float,
    actual_next_low: float,
    tol: float = 1e-3,
) -> str:
    """Evaluates whether `actual_next_close` (and `actual_next_low` if intraday-gated) fires `trigger`."""
    if not trigger.can_trigger_next_day:
        return "HOLD" if curr_pos == 1 else "WAIT"

    # Check intraday low gate (only relevant for S06 when past 4 days did not touch SMA50)
    if trigger.intraday_low_required and trigger.intraday_low_threshold is not None:
        if actual_next_low > trigger.intraday_low_threshold + tol:
            return "HOLD" if curr_pos == 1 else "WAIT"

    fired = False
    if trigger.comparison == "RANGE" and trigger.price_range is not None:
        lo, hi = trigger.price_range
        above_lo = True if lo is None else (actual_next_close > lo - tol)
        below_hi = True if hi is None else (actual_next_close <= hi + tol)
        fired = above_lo and below_hi
    elif trigger.trigger_price is not None:
        tp = trigger.trigger_price
        if trigger.comparison == ">":
            fired = actual_next_close > tp
        elif trigger.comparison == ">=":
            fired = actual_next_close >= tp - tol
        elif trigger.comparison == "<":
            fired = actual_next_close < tp
        elif trigger.comparison == "<=":
            fired = actual_next_close <= tp + tol

    if fired:
        return "BUY" if curr_pos == 0 else "SELL"
    return "HOLD" if curr_pos == 1 else "WAIT"


def run_historical_replay_validation(
    strategy_code: str,
    df: pd.DataFrame,
    ticker: str,
    replay_days: int = 60,
    strategy=None,
) -> ReplayValidationResult:
    """
    Strictly no-lookahead historical replay validation for single-stock strategies (S01, S02, S03, S05, S06).

    For each historical day `t` in the last `replay_days` pairs `(t, t+1)`:
    1. Exposes ONLY `df.iloc[:t+1]` to the preview module (`solve_single_stock_forward_signal`).
    2. Feeds the actual next bar `df.iloc[t+1]` into `ForwardScenarioEngine.evaluate_next_day_price`
       and verifies 100% parity against the original full-history strategy signal at `t+1`.
    3. Tests whether the solved `TriggerCondition` alone correctly predicts the actual `BUY`/`SELL`/`HOLD`/`WAIT`
       transition when given the real `t+1` bar.
    """
    code = strategy_code.upper()
    if strategy is None:
        strategy = create_single_stock_strategy(code)

    full_sig = strategy.generate_signals(df)
    n = len(df)
    if n < 5:
        raise ValueError("Not enough historical bars for replay validation.")

    # Need t in [start_idx, n - 2] so t+1 exists
    max_pairs = n - 2
    num_days = max(1, min(int(replay_days), max_pairs))
    start_idx = n - 1 - num_days  # t goes from start_idx .. n-2

    scenario_matches = 0
    threshold_matches = 0
    buy_actual = 0
    buy_matched = 0
    sell_actual = 0
    sell_matched = 0
    records: List[Dict[str, Any]] = []

    for t in range(start_idx, n - 1):
        df_as_of_t = df.iloc[: t + 1]
        next_bar = df.iloc[t + 1]
        next_dt = df.index[t + 1]
        as_of_dt = df.index[t]

        actual_prev_pos = int(full_sig["Signal"].iloc[t])
        actual_next_pos = int(full_sig["Signal"].iloc[t + 1])
        actual_action = classify_transition(actual_prev_pos, actual_next_pos)

        # Step A: Run forward preview using strictly data up to t
        report_t = solve_single_stock_forward_signal(code, df_as_of_t, ticker=ticker, strategy=strategy)

        # Step B: Feed real next bar into ForwardScenarioEngine
        scen_pos, _, scen_action, _ = ForwardScenarioEngine.evaluate_next_day_price(
            df_as_of_t,
            strategy,
            test_close=float(next_bar["Close"]),
            test_low=float(next_bar["Low"]),
            test_high=float(next_bar["High"]),
            test_open=float(next_bar["Open"]),
            next_date=next_dt,
        )
        scen_ok = (scen_pos == actual_next_pos) and (scen_action == actual_action)
        if scen_ok:
            scenario_matches += 1

        # Step C: Evaluate solved TriggerCondition against real next bar
        primary_trig = report_t.primary_trigger
        assert primary_trig is not None
        thresh_action = _predict_from_trigger(
            curr_pos=actual_prev_pos,
            trigger=primary_trig,
            actual_next_close=float(next_bar["Close"]),
            actual_next_low=float(next_bar["Low"]),
        )
        thresh_ok = (thresh_action == actual_action)
        if thresh_ok:
            threshold_matches += 1

        if actual_action == "BUY":
            buy_actual += 1
            if scen_ok and thresh_ok:
                buy_matched += 1
        elif actual_action == "SELL":
            sell_actual += 1
            if scen_ok and thresh_ok:
                sell_matched += 1

        records.append(
            {
                "决策基准日 (T)": as_of_dt.strftime("%Y-%m-%d"),
                "验证交易日 (T+1)": next_dt.strftime("%Y-%m-%d"),
                "T 日收盘价": f"${float(df['Close'].iloc[t]):,.2f}",
                "T+1 真实收盘": f"${float(next_bar['Close']):,.2f}",
                "T 日预测触发条件": primary_trig.format_trigger_label(),
                "原始回测真实动作": actual_action,
                "情景引擎重放": scen_action,
                "阈值反推判定": thresh_action,
                "一致性校验": "✅ 一致" if (scen_ok and thresh_ok) else "❌ 偏差",
            }
        )

    total = len(records)
    scen_acc = (scenario_matches / total * 100.0) if total > 0 else 100.0
    thresh_acc = (threshold_matches / total * 100.0) if total > 0 else 100.0

    notes = [
        f"严格无前视重放：对过去 {total} 个连续交易日逐日截断历史至 T 日，隐藏 T+1 及之后所有行情。",
        f"情景引擎 (ForwardScenarioEngine) 严格调用原始策略逻辑，与完整历史回测的一致率为 **{scen_acc:.2f}%** ({scenario_matches}/{total})。",
        f"临界阈值求解器 (SignalThresholdSolver) 预测与真实回测动作一致率为 **{thresh_acc:.2f}%** ({threshold_matches}/{total})。",
    ]

    return ReplayValidationResult(
        strategy_code=code,
        ticker_or_pool=ticker,
        total_days_tested=total,
        scenario_exact_matches=scenario_matches,
        threshold_exact_matches=threshold_matches,
        mismatches=total - min(scenario_matches, threshold_matches),
        scenario_accuracy_pct=scen_acc,
        threshold_accuracy_pct=thresh_acc,
        buy_transitions_actual=buy_actual,
        buy_transitions_matched=buy_matched,
        sell_transitions_actual=sell_actual,
        sell_transitions_matched=sell_matched,
        details_df=pd.DataFrame(records),
        notes=notes,
    )


def run_portfolio_replay_validation(
    strategy_code: str,
    prices_df: pd.DataFrame,
    universe: List[str],
    replay_months: int = 12,
) -> ReplayValidationResult:
    """
    No-lookahead rebalance-point replay validation for portfolio strategies (S04, S07, S08, S09).
    Verifies that computing target weights strictly with data up to each historical rebalance date `t`
    produces the exact same target allocation as the full-history backtest!
    """
    code = strategy_code.upper()
    strategy = create_portfolio_strategy(code, universe)

    if code == "S07":
        # Replay month-end drift trigger parity against ThresholdRebalancing.simulate
        full_res = strategy.simulate(prices_df=prices_df)
        dates = pd.Series(prices_df.index, index=prices_df.index)
        month_ends = list(dates.groupby([dates.dt.year, dates.dt.month]).last().values)
        # Exclude very last date if incomplete month
        check_dates = month_ends[-replay_months - 1 : -1] if len(month_ends) > 1 else month_ends

        matches = 0
        records = []
        rebal_date_set = {pd.Timestamp(ev.date) for ev in full_res.rebalance_events}

        for dt in check_dates:
            dt_ts = pd.Timestamp(dt)
            loc = prices_df.index.get_loc(dt_ts)
            if loc + 1 >= len(prices_df):
                continue
            next_dt = prices_df.index[loc + 1]
            sub_df = prices_df.iloc[: loc + 1]
            sub_res = strategy.simulate(prices_df=sub_df)

            # Check if sub_res weights on dt breach bands
            w_t = sub_res.weights_history.iloc[-1].to_dict()
            pred_trigger = any(
                (w_t.get(a, 0.0) > b["upper"] or w_t.get(a, 0.0) < b["lower"])
                for a, b in strategy.bands.items()
                if pd.notna(sub_df[a].iloc[-1]) and sub_df[a].iloc[-1] > 0
            )
            actual_rebal = pd.Timestamp(next_dt) in rebal_date_set
            ok = (pred_trigger == actual_rebal)
            if ok:
                matches += 1

            records.append(
                {
                    "检验月末日 (T)": dt_ts.strftime("%Y-%m-%d"),
                    "执行日 (T+1)": pd.Timestamp(next_dt).strftime("%Y-%m-%d"),
                    "无前视重放判定": "REBALANCE" if pred_trigger else "HOLD",
                    "完整回测真实执行": "REBALANCE" if actual_rebal else "HOLD",
                    "一致性校验": "✅ 一致" if ok else "❌ 偏差",
                }
            )

        total = len(records)
        acc = (matches / total * 100.0) if total > 0 else 100.0
        return ReplayValidationResult(
            strategy_code=code,
            ticker_or_pool=f"组合 ({len(universe)} 只)",
            total_days_tested=total,
            scenario_exact_matches=matches,
            threshold_exact_matches=matches,
            mismatches=total - matches,
            scenario_accuracy_pct=acc,
            threshold_accuracy_pct=acc,
            buy_transitions_actual=sum(1 for r in records if r["完整回测真实执行"] == "REBALANCE"),
            buy_transitions_matched=sum(1 for r in records if r["完整回测真实执行"] == "REBALANCE" and r["一致性校验"].startswith("✅")),
            sell_transitions_actual=0,
            sell_transitions_matched=0,
            details_df=pd.DataFrame(records),
            notes=[
                f"对最近 {total} 个月末检验日执行严格截断重放，核验容忍带越界判定与真实再平衡事件一致率：**{acc:.2f}%**。"
            ],
        )

    # S04 / S08 / S09: compute_target_weights parity check
    full_w = strategy.compute_target_weights(prices_df)
    if full_w.empty:
        return ReplayValidationResult(
            strategy_code=code,
            ticker_or_pool=f"组合 ({len(prices_df.columns)} 只)",
            total_days_tested=0,
            scenario_exact_matches=0,
            threshold_exact_matches=0,
            mismatches=0,
            scenario_accuracy_pct=100.0,
            threshold_accuracy_pct=100.0,
            buy_transitions_actual=0,
            buy_transitions_matched=0,
            sell_transitions_actual=0,
            sell_transitions_matched=0,
            details_df=pd.DataFrame(),
            notes=["当前时间窗口内尚无完整历史调仓节点可供重放。"],
        )

    check_dates = list(full_w.index[-replay_months:])
    matches = 0
    records = []
    for dt in check_dates:
        dt_ts = pd.Timestamp(dt)
        loc = prices_df.index.get_loc(dt_ts)
        sub_df = prices_df.iloc[: loc + 1]
        sub_w = strategy.compute_target_weights(sub_df)

        full_row = full_w.loc[dt_ts]
        if dt_ts in sub_w.index:
            sub_row = sub_w.loc[dt_ts]
            ok = bool(np.allclose(full_row.values.astype(float), sub_row.values.astype(float), atol=1e-6))
            pred_picks = ", ".join([a for a, v in sub_row.items() if v > 0]) or "100% 现金"
        else:
            ok = False
            pred_picks = "未生成"

        actual_picks = ", ".join([a for a, v in full_row.items() if v > 0]) or "100% 现金"
        if ok:
            matches += 1

        records.append(
            {
                "调仓检验日 (T)": dt_ts.strftime("%Y-%m-%d"),
                "无前视重放目标组合": pred_picks,
                "完整回测真实目标组合": actual_picks,
                "一致性校验": "✅ 一致" if ok else "❌ 偏差",
            }
        )

    total = len(records)
    acc = (matches / total * 100.0) if total > 0 else 100.0
    notes = [
        f"对最近 {total} 个历史调仓检验日执行严格截断（仅使用截至 T 日数据），目标权重完全一致率：**{acc:.2f}%** ({matches}/{total})。"
    ]
    if code == "S08":
        notes.append("说明：此处验证的是当前回测代码逻辑的无前视一致性；实盘仍需接入真实 Point-in-Time 季度财报数据。")

    return ReplayValidationResult(
        strategy_code=code,
        ticker_or_pool=f"组合 ({len(prices_df.columns)} 只)",
        total_days_tested=total,
        scenario_exact_matches=matches,
        threshold_exact_matches=matches,
        mismatches=total - matches,
        scenario_accuracy_pct=acc,
        threshold_accuracy_pct=acc,
        buy_transitions_actual=total,
        buy_transitions_matched=matches,
        sell_transitions_actual=0,
        sell_transitions_matched=0,
        details_df=pd.DataFrame(records),
        notes=notes,
    )
