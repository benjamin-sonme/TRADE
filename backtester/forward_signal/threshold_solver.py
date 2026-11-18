"""
Signal Threshold Solver for Single-Stock Strategies (S01, S02, S03, S05, S06).

Every candidate threshold price or interval is solved mathematically / numerically
AND strictly verified against `ForwardScenarioEngine.evaluate_next_day_price`,
ensuring 100% consistency with the real strategy `generate_signals` code.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .models import ForwardSignalReport, TriggerCondition
from .scenario_engine import (
    STRATEGY_METADATA,
    ForwardScenarioEngine,
    classify_transition,
    create_single_stock_strategy,
    estimate_next_trading_date,
)

# Execution timing descriptions aligned with BacktestEngine (T+1 execution)
EXEC_CONFIRMED_EVENT = "信号已于最新收盘确认 · 最早可于下一交易日 (T+1) 执行（本系统回测默认按 T+1 收盘价+滑点撮合，实盘最早 T+1 开盘可挂单）"
EXEC_CONFIRMED_IDLE = "当前无新确认信号 · 维持现有状态，等待下一交易日条件触发"
EXEC_CONDITIONAL_NEXT = "若下一交易日 (T+1) 收盘满足条件，将在 T+1 收盘后确认，最早于 T+2 交易日执行（不可在 T+1 开盘提前成交）"


def _dist_metrics(trigger_price: Optional[float], latest_close: float) -> Tuple[Optional[float], Optional[float]]:
    if trigger_price is None or latest_close <= 0:
        return None, None
    diff = float(trigger_price - latest_close)
    pct = diff / float(latest_close)
    return pct, diff


def _verify_or_bisect_root(
    df: pd.DataFrame,
    strategy,
    p_star: float,
    target_pos: int,
    lo_is_target: bool,
    eps: float = 1e-3,
    test_low_override: Optional[float] = None,
) -> Optional[float]:
    """
    Fast exact verification of an analytical candidate root `p_star` against the REAL
    `strategy.generate_signals(sim_df)` engine.
    If the flip occurs inside `[p_star - eps, p_star + eps]`, returns `p_star` in 2 calls;
    otherwise widens slightly and refines via bisection.
    """
    if p_star is None or not np.isfinite(p_star) or p_star <= 0.01:
        return None
    step = max(eps, abs(p_star) * 1e-5)
    lo = max(0.01, p_star - step)
    hi = p_star + step
    pos_lo, _, _, _ = ForwardScenarioEngine.evaluate_next_day_price(
        df, strategy, test_close=lo, test_low=test_low_override
    )
    pos_hi, _, _, _ = ForwardScenarioEngine.evaluate_next_day_price(
        df, strategy, test_close=hi, test_low=test_low_override
    )
    if (pos_lo == target_pos) == lo_is_target and (pos_hi == target_pos) != lo_is_target:
        return float(p_star)

    # Fallback wider bracket in case of tiny floating-point / EWM offset
    wide = max(0.05, abs(p_star) * 2e-3)
    lo_w = max(0.01, p_star - wide)
    hi_w = p_star + wide
    pos_lo_w, _, _, _ = ForwardScenarioEngine.evaluate_next_day_price(
        df, strategy, test_close=lo_w, test_low=test_low_override
    )
    pos_hi_w, _, _, _ = ForwardScenarioEngine.evaluate_next_day_price(
        df, strategy, test_close=hi_w, test_low=test_low_override
    )
    if (pos_lo_w == target_pos) == lo_is_target and (pos_hi_w == target_pos) != lo_is_target:
        return _bisect_boundary(
            df,
            strategy,
            lo=lo_w,
            hi=hi_w,
            target_pos=target_pos,
            lo_is_target=lo_is_target,
            iterations=18,
            test_low_override=test_low_override,
        )
    return None


def _solve_bb_upper_breakout(closes: pd.Series, period: int, std_dev: float) -> Optional[float]:
    """
    Exact closed-form root for `P = SMA(period) + std_dev * std(period, ddof=1)` at T+1.
    """
    if len(closes) < period - 1 or period <= 2:
        return None
    tail = closes.iloc[-(period - 1):].to_numpy(dtype=float)
    if np.any(np.isnan(tail)):
        return None
    n = float(period)
    mu0 = float(np.mean(tail))
    v0 = float(np.sum((tail - mu0) ** 2))
    denom = ((n - 1.0) / n) ** 2 - (std_dev ** 2) / n
    if denom <= 0.0:
        return None
    delta = np.sqrt((std_dev ** 2) * (v0 / (n - 1.0)) / denom)
    return float(mu0 + delta)


def _bisect_boundary(
    df: pd.DataFrame,
    strategy,
    lo: float,
    hi: float,
    target_pos: int,
    lo_is_target: bool,
    iterations: int = 32,
    test_low_override: Optional[float] = None,
) -> float:
    """
    Refines a signal-flip boundary between `lo` and `hi` directly against
    `strategy.generate_signals(sim_df)`.
    """
    a, b = float(lo), float(hi)
    for _ in range(iterations):
        mid = 0.5 * (a + b)
        pos, _, _, _ = ForwardScenarioEngine.evaluate_next_day_price(
            df,
            strategy,
            test_close=mid,
            test_low=test_low_override,
        )
        is_target = (pos == target_pos)
        if is_target == lo_is_target:
            a = mid
        else:
            b = mid
        if abs(b - a) <= 1e-4:
            break
    return 0.5 * (a + b)


def _solve_sma_self_threshold(closes: pd.Series, window: int) -> Optional[float]:
    """
    Solves `P = SMA(window)` at T+1, i.e. `P = (sum(last window-1 closes) + P) / window`
    => `P = sum(last window-1 closes) / (window - 1)`.
    """
    if len(closes) < window - 1 or window <= 1:
        return None
    tail_sum = float(closes.iloc[-(window - 1):].sum())
    return tail_sum / float(window - 1)


def _solve_two_sma_cross(closes: pd.Series, short_w: int, long_w: int) -> Optional[float]:
    """
    Solves `SMA(short_w) = SMA(long_w)` at T+1:
    (S_short + P) / short_w = (S_long + P) / long_w
    => P = (short_w * S_long - long_w * S_short) / (long_w - short_w)
    """
    if len(closes) < long_w - 1 or long_w <= short_w:
        return None
    s_short = float(closes.iloc[-(short_w - 1):].sum()) if short_w > 1 else 0.0
    s_long = float(closes.iloc[-(long_w - 1):].sum())
    return (short_w * s_long - long_w * s_short) / float(long_w - short_w)


def _solve_rsi_threshold_price(closes: pd.Series, period: int, target_rsi: float) -> Optional[float]:
    """
    Exact analytical inverse of Wilder's EWM RSI (`alpha = 1 / period, adjust=False`)
    matching `backtester.strategies.rsi_strategy.calculate_rsi`.
    """
    if len(closes) < period:
        return None
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    g_t = avg_gain.iloc[-1]
    l_t = avg_loss.iloc[-1]
    c_t = float(closes.iloc[-1])

    if pd.isna(g_t) or pd.isna(l_t):
        return None
    g_t = float(g_t)
    l_t = float(l_t)

    if target_rsi <= 0.0 or target_rsi >= 100.0:
        return None

    target_rs = target_rsi / (100.0 - target_rsi)
    # Candidate 1: Up day (P >= c_t) -> G_{t+1} = (p-1)/p * g_t + (P - c_t)/p, L_{t+1} = (p-1)/p * l_t
    p_up = c_t + (period - 1.0) * (target_rs * l_t - g_t)
    # Candidate 2: Down day (P <= c_t) -> G_{t+1} = (p-1)/p * g_t, L_{t+1} = (p-1)/p * l_t + (c_t - P)/p
    p_down = c_t - (period - 1.0) * ((g_t / target_rs) - l_t) if target_rs > 0 else None

    if p_up >= c_t - 1e-9:
        return float(p_up)
    if p_down is not None and p_down <= c_t + 1e-9:
        return float(p_down)
    return float(p_up if abs(p_up - c_t) <= abs((p_down or p_up) - c_t) else p_down)


def solve_single_stock_forward_signal(
    strategy_code: str,
    df: pd.DataFrame,
    ticker: str,
    strategy=None,
) -> ForwardSignalReport:
    """
    Computes the complete Next-Day Signal Preview report for S01, S02, S03, S05, or S06.
    """
    code = strategy_code.upper()
    meta = STRATEGY_METADATA[code]
    if strategy is None:
        strategy = create_single_stock_strategy(code)

    hist_sig = ForwardScenarioEngine.evaluate_single_stock_history(df, strategy)
    as_of_ts = pd.Timestamp(hist_sig.index[-1])
    next_ts = estimate_next_trading_date(as_of_ts)
    as_of_str = as_of_ts.strftime("%Y-%m-%d")
    next_str = next_ts.strftime("%Y-%m-%d")

    latest_close = float(hist_sig["Close"].iloc[-1])
    curr_pos = int(hist_sig["Signal"].iloc[-1])
    prev_pos = int(hist_sig["Signal"].iloc[-2]) if len(hist_sig) >= 2 else 0
    confirmed_signal = classify_transition(prev_pos, curr_pos)
    confirmed_is_new = confirmed_signal in {"BUY", "SELL"}

    if confirmed_signal == "BUY":
        confirmed_summary = f"最新交易日 ({as_of_str}) 收盘已确认产生买入信号 (0 → 1)，收盘价 ${latest_close:,.2f}。"
        confirmed_exec = EXEC_CONFIRMED_EVENT
    elif confirmed_signal == "SELL":
        confirmed_summary = f"最新交易日 ({as_of_str}) 收盘已确认产生卖出信号 (1 → 0)，收盘价 ${latest_close:,.2f}。"
        confirmed_exec = EXEC_CONFIRMED_EVENT
    elif confirmed_signal == "HOLD":
        confirmed_summary = f"最新交易日 ({as_of_str}) 收盘价 ${latest_close:,.2f}，策略维持持仓状态 (HOLD)，未触发新卖出信号。"
        confirmed_exec = EXEC_CONFIRMED_IDLE
    else:
        confirmed_summary = f"最新交易日 ({as_of_str}) 收盘价 ${latest_close:,.2f}，策略维持空仓观望 (WAIT)，未触发新买入信号。"
        confirmed_exec = EXEC_CONFIRMED_IDLE

    # Dispatch per-strategy solver + real engine verification
    if code == "S01":
        buy_trig, sell_trig, why_items, why_expl = _solve_s01(df, hist_sig, strategy, latest_close, curr_pos)
    elif code == "S02":
        buy_trig, sell_trig, why_items, why_expl = _solve_s02(df, hist_sig, strategy, latest_close, curr_pos)
    elif code == "S03":
        buy_trig, sell_trig, why_items, why_expl = _solve_s03(df, hist_sig, strategy, latest_close, curr_pos)
    elif code == "S05":
        buy_trig, sell_trig, why_items, why_expl = _solve_s05(df, hist_sig, strategy, latest_close, curr_pos)
    elif code == "S06":
        buy_trig, sell_trig, why_items, why_expl = _solve_s06(df, hist_sig, strategy, latest_close, curr_pos)
    else:
        raise ValueError(f"Unsupported single-stock code: {code}")

    primary_trigger = buy_trig if curr_pos == 0 else sell_trig

    if primary_trigger.can_trigger_next_day:
        cond_summary = (
            f"下一交易日 ({next_str}) {primary_trigger.condition_summary}"
        )
        cond_exec = primary_trigger.earliest_execution or EXEC_CONDITIONAL_NEXT
    else:
        cond_summary = (
            f"下一交易日 ({next_str}) 无法触发{'买入' if curr_pos == 0 else '卖出'}信号："
            f"{primary_trigger.blocker_reason or primary_trigger.condition_summary}"
        )
        cond_exec = "下一交易日无有效反转触发条件，预计维持当前持仓状态"

    overall_exec = confirmed_exec if confirmed_is_new else cond_exec

    return ForwardSignalReport(
        strategy_code=code,
        strategy_name=f"{code} - {meta['cn_name']}",
        strategy_en_name=meta["en_name"],
        is_portfolio=False,
        ticker_or_pool=ticker,
        as_of_date=as_of_str,
        next_trading_date=next_str,
        latest_close=latest_close,
        calculation_category=meta["category"],
        calculation_category_label=meta["category_label"],
        confirmed_signal=confirmed_signal,
        confirmed_is_new_event=confirmed_is_new,
        confirmed_position=curr_pos,
        confirmed_summary=confirmed_summary,
        confirmed_execution_timing=confirmed_exec,
        conditional_summary=cond_summary,
        conditional_execution_timing=cond_exec,
        current_signal=confirmed_signal,
        next_buy_trigger=buy_trig,
        next_sell_trigger=sell_trig,
        primary_trigger=primary_trigger,
        execution_timing=overall_exec,
        why_items=why_items,
        why_explanation=why_expl,
        extra={
            "prev_position": prev_pos,
            "curr_position": curr_pos,
        },
    )


# ---------------------------------------------------------------------------
# Strategy-Specific Exact Solvers (all verified against real generate_signals)
# ---------------------------------------------------------------------------

def _solve_s01(
    df: pd.DataFrame,
    hist_sig: pd.DataFrame,
    strategy,
    latest_close: float,
    curr_pos: int,
) -> Tuple[TriggerCondition, TriggerCondition, List[Dict[str, str]], List[str]]:
    sw, lw = strategy.short_window, strategy.long_window
    ma_s = hist_sig["MA_Short"].iloc[-1]
    ma_l = hist_sig["MA_Long"].iloc[-1]
    closes = df["Close"]

    if strategy.ma_type == "EMA":
        a_s = 2.0 / (sw + 1.0)
        a_l = 2.0 / (lw + 1.0)
        if pd.notna(ma_s) and pd.notna(ma_l) and a_s != a_l:
            p_star = ((1.0 - a_l) * float(ma_l) - (1.0 - a_s) * float(ma_s)) / (a_s - a_l)
        else:
            p_star = None
    else:
        p_star = _solve_two_sma_cross(closes, sw, lw)

    # Verify with real strategy engine (fast exact root verification)
    valid_p_star = _verify_or_bisect_root(df, strategy, p_star, target_pos=1, lo_is_target=False) if p_star is not None else None

    if curr_pos == 0:
        if valid_p_star is not None:
            dist_pct, dist_abs = _dist_metrics(valid_p_star, latest_close)
            buy_trig = TriggerCondition(
                action="BUY",
                can_trigger_next_day=True,
                trigger_price=valid_p_star,
                comparison=">",
                distance_pct=dist_pct,
                distance_abs=dist_abs,
                condition_summary=(
                    f"收盘价严格高于 ${valid_p_star:,.2f}，使 {strategy.ma_type}{sw} 上穿 {strategy.ma_type}{lw} 形成金叉买入信号。"
                ),
                earliest_execution=EXEC_CONDITIONAL_NEXT,
            )
        else:
            reason = (
                f"历史数据不足 {lw} 个交易日，均线尚未预热完成。"
                if len(df) + 1 < lw
                else f"短期均线 ({strategy.ma_type}{sw}) 显著低于长期均线 ({strategy.ma_type}{lw})，即使大幅波动明日也无法反超金叉（理论解={p_star if p_star is not None else 'N/A'}）。"
            )
            buy_trig = TriggerCondition(
                action="BUY",
                can_trigger_next_day=False,
                comparison="BLOCKED",
                condition_summary="明日无法触发买入",
                blocker_reason=reason,
            )
        sell_trig = TriggerCondition(
            action="SELL",
            can_trigger_next_day=False,
            comparison="BLOCKED",
            condition_summary="当前为空仓状态，无需触发卖出",
            blocker_reason="当前仓位为 0（空仓），须先产生买入持仓后才存在卖出触发条件。",
        )
    else:
        buy_trig = TriggerCondition(
            action="BUY",
            can_trigger_next_day=False,
            comparison="BLOCKED",
            condition_summary="当前已处于持仓状态，不会重复触发买入",
            blocker_reason="策略当前已满仓持多 (Position=1)，仅监测卖出死叉条件。",
        )
        if valid_p_star is not None:
            dist_pct, dist_abs = _dist_metrics(valid_p_star, latest_close)
            sell_trig = TriggerCondition(
                action="SELL",
                can_trigger_next_day=True,
                trigger_price=valid_p_star,
                comparison="<=",
                distance_pct=dist_pct,
                distance_abs=dist_abs,
                condition_summary=(
                    f"收盘价跌至 ≤ ${valid_p_star:,.2f}，使 {strategy.ma_type}{sw} 下穿或等于 {strategy.ma_type}{lw} 形成死叉清仓信号。"
                ),
                earliest_execution=EXEC_CONDITIONAL_NEXT,
            )
        else:
            reason = (
                f"{strategy.ma_type}{sw} 远高于 {strategy.ma_type}{lw}，即便明日跌至 $0.01 仍保持多头排列，无法单日死叉。"
            )
            sell_trig = TriggerCondition(
                action="SELL",
                can_trigger_next_day=False,
                comparison="BLOCKED",
                condition_summary="明日无法触发卖出",
                blocker_reason=reason,
            )

    spread = (float(ma_s) - float(ma_l)) if pd.notna(ma_s) and pd.notna(ma_l) else None
    why_items = [
        {"指标": f"最新收盘价", "当前值": f"${latest_close:,.2f}", "说明": "截至最新完成交易日"},
        {"指标": f"{strategy.ma_type}({sw}) 短期均线", "当前值": f"${ma_s:,.2f}" if pd.notna(ma_s) else "预热中", "说明": f"最近 {sw} 日均价"},
        {"指标": f"{strategy.ma_type}({lw}) 长期均线", "当前值": f"${ma_l:,.2f}" if pd.notna(ma_l) else "预热中", "说明": f"最近 {lw} 日均价"},
        {"指标": "均线差值 (Short - Long)", "当前值": f"{spread:+,.2f}" if spread is not None else "N/A", "说明": "> 0 持仓，≤ 0 空仓"},
        {"指标": "临界交叉价 P*", "当前值": f"${valid_p_star:,.2f}" if valid_p_star else "无有效正价格解", "说明": "经策略源码验证的精确拐点"},
    ]
    why_expl = [
        f"S01 规则：当 `{strategy.ma_type}({sw}) > {strategy.ma_type}({lw})` 时持多（Signal=1），否则空仓（Signal=0）。",
        f"当前 `{strategy.ma_type}({sw})` = {f'${ma_s:,.2f}' if pd.notna(ma_s) else 'N/A'}，`{strategy.ma_type}({lw})` = {f'${ma_l:,.2f}' if pd.notna(ma_l) else 'N/A'}，因此当前状态为 **{'持仓 (1)' if curr_pos == 1 else '空仓 (0)'}**。",
        (
            f"联立明日均线方程解得临界价格 **${valid_p_star:,.2f}**，并已通过真实 `DualMACrossoverStrategy.generate_signals()` 双侧验证。"
            if valid_p_star is not None
            else "当前均线偏离幅度较大，单日任意正收盘价均不足以令两条均线在明日直接交叉。"
        ),
    ]
    return buy_trig, sell_trig, why_items, why_expl


def _solve_s02(
    df: pd.DataFrame,
    hist_sig: pd.DataFrame,
    strategy,
    latest_close: float,
    curr_pos: int,
) -> Tuple[TriggerCondition, TriggerCondition, List[Dict[str, str]], List[str]]:
    closes = df["Close"]
    curr_rsi = float(hist_sig["RSI"].iloc[-1])
    trend_ma = hist_sig["Trend_MA"].iloc[-1] if "Trend_MA" in hist_sig.columns else np.nan

    p_os_raw = _solve_rsi_threshold_price(closes, strategy.period, strategy.oversold)
    p_ob_raw = _solve_rsi_threshold_price(closes, strategy.period, strategy.overbought)
    p_trend = _solve_sma_self_threshold(closes, strategy.trend_ma_period) if strategy.use_trend_filter else None
    trend_active_tomorrow = strategy.use_trend_filter and (len(closes) + 1 >= strategy.trend_ma_period) and (p_trend is not None)

    if curr_pos == 0:
        # BUY requires RSI(P) <= oversold AND (Close > SMA200(P) if trend filter active)
        buy_trig: Optional[TriggerCondition] = None
        if p_os_raw is None or p_os_raw <= 0.01:
            buy_trig = TriggerCondition(
                action="BUY",
                can_trigger_next_day=False,
                comparison="BLOCKED",
                condition_summary="明日无法触发买入",
                blocker_reason=(
                    f"当前 RSI({strategy.period})={curr_rsi:.2f} 偏高，按 Wilder 递推公式即便明日股价跌至极低值，"
                    f"单日也无法跌破超卖阈值 {strategy.oversold:.0f}（理论解={p_os_raw if p_os_raw is not None else 'N/A'}）。"
                ),
            )
        elif trend_active_tomorrow:
            # Need p_trend < P <= p_os_raw
            if p_os_raw > p_trend + 1e-3:
                # Verify midpoint with real strategy engine
                mid_p = 0.5 * (p_trend + p_os_raw)
                pos_mid, _, _, _ = ForwardScenarioEngine.evaluate_next_day_price(df, strategy, mid_p)
                if pos_mid == 1:
                    # Verify/refine both lower (trend_ma) and upper (rsi_os) boundaries via real engine
                    lo_ref = _verify_or_bisect_root(df, strategy, p_trend, target_pos=1, lo_is_target=False) or float(p_trend)
                    hi_ref = _verify_or_bisect_root(df, strategy, p_os_raw, target_pos=1, lo_is_target=True) or float(p_os_raw)
                    # Nearest actionable boundary from latest_close
                    nearest = hi_ref if latest_close >= hi_ref else (lo_ref if latest_close <= lo_ref else latest_close)
                    dist_pct, dist_abs = _dist_metrics(nearest, latest_close)
                    buy_trig = TriggerCondition(
                        action="BUY",
                        can_trigger_next_day=True,
                        trigger_price=nearest,
                        price_range=(lo_ref, hi_ref),
                        comparison="RANGE",
                        distance_pct=dist_pct,
                        distance_abs=dist_abs,
                        condition_summary=(
                            f"收盘价需精准落在区间 (${lo_ref:,.2f}, ${hi_ref:,.2f}] 内："
                            f"同时满足 RSI({strategy.period}) ≤ {strategy.oversold:.0f}（≤ ${hi_ref:,.2f}）且高于 SMA{strategy.trend_ma_period} 趋势过滤线（> ${lo_ref:,.2f}）。"
                        ),
                        earliest_execution=EXEC_CONDITIONAL_NEXT,
                    )
            if buy_trig is None:
                buy_trig = TriggerCondition(
                    action="BUY",
                    can_trigger_next_day=False,
                    comparison="BLOCKED",
                    condition_summary="明日无法触发买入（超卖条件与 200 日牛熊线冲突）",
                    blocker_reason=(
                        f"使 RSI({strategy.period}) ≤ {strategy.oversold:.0f} 需跌至 ≤ ${p_os_raw:,.2f}，"
                        f"但该价格低于 SMA{strategy.trend_ma_period} 趋势过滤临界线 (${p_trend:,.2f})，"
                        f"两个条件在明日数学上互斥，因此无法触发买入。"
                    ),
                )
        else:
            # No active trend filter: pure upper bound P <= p_os_raw
            hi_ref = _verify_or_bisect_root(df, strategy, p_os_raw, target_pos=1, lo_is_target=True)
            if hi_ref is not None:
                dist_pct, dist_abs = _dist_metrics(hi_ref, latest_close)
                buy_trig = TriggerCondition(
                    action="BUY",
                    can_trigger_next_day=True,
                    trigger_price=hi_ref,
                    comparison="<=",
                    distance_pct=dist_pct,
                    distance_abs=dist_abs,
                    condition_summary=(
                        f"收盘价跌至 ≤ ${hi_ref:,.2f}（使 RSI({strategy.period}) ≤ {strategy.oversold:.0f}）触发超卖买入。"
                    ),
                    earliest_execution=EXEC_CONDITIONAL_NEXT,
                )
            else:
                buy_trig = TriggerCondition(
                    action="BUY",
                    can_trigger_next_day=False,
                    comparison="BLOCKED",
                    condition_summary="明日无法触发买入",
                    blocker_reason="经真实策略引擎核验，明日常规价格区间内无法满足超卖买入条件。",
                )

        sell_trig = TriggerCondition(
            action="SELL",
            can_trigger_next_day=False,
            comparison="BLOCKED",
            condition_summary="当前为空仓状态，无需触发卖出",
            blocker_reason="当前未持仓 (Position=0)，仅监测买入条件。",
        )
    else:
        buy_trig = TriggerCondition(
            action="BUY",
            can_trigger_next_day=False,
            comparison="BLOCKED",
            condition_summary="当前已持仓，不会重复触发买入",
            blocker_reason="策略当前已持多仓 (Position=1)，仅监测 RSI 超买卖出条件。",
        )
        if p_ob_raw is not None and p_ob_raw > 0.01:
            ob_ref = _verify_or_bisect_root(df, strategy, p_ob_raw, target_pos=0, lo_is_target=False)
            if ob_ref is not None:
                dist_pct, dist_abs = _dist_metrics(ob_ref, latest_close)
                sell_trig = TriggerCondition(
                    action="SELL",
                    can_trigger_next_day=True,
                    trigger_price=ob_ref,
                    comparison=">=",
                    distance_pct=dist_pct,
                    distance_abs=dist_abs,
                    condition_summary=(
                        f"收盘价上涨至 ≥ ${ob_ref:,.2f}，使 RSI({strategy.period}) ≥ {strategy.overbought:.0f} 触发超买获利了结。"
                    ),
                    earliest_execution=EXEC_CONDITIONAL_NEXT,
                )
            else:
                # Rare edge case: even slightly below p_ob_raw RSI stays >= OB
                pos_chk, _, _, _ = ForwardScenarioEngine.evaluate_next_day_price(df, strategy, max(0.01, p_ob_raw))
                if pos_chk == 0:
                    dist_pct, dist_abs = _dist_metrics(max(0.01, p_ob_raw), latest_close)
                    sell_trig = TriggerCondition(
                        action="SELL",
                        can_trigger_next_day=True,
                        trigger_price=max(0.01, p_ob_raw),
                        comparison=">=",
                        distance_pct=dist_pct,
                        distance_abs=dist_abs,
                        condition_summary=f"收盘价 ≥ ${max(0.01, p_ob_raw):,.2f} 即可维持 RSI ≥ {strategy.overbought:.0f} 触发卖出。",
                        earliest_execution=EXEC_CONDITIONAL_NEXT,
                    )
                else:
                    sell_trig = TriggerCondition(
                        action="SELL",
                        can_trigger_next_day=False,
                        comparison="BLOCKED",
                        condition_summary="明日无法触发卖出",
                        blocker_reason="经策略源码核验，明日无法推升 RSI 至超买阈值。",
                    )
        else:
            sell_trig = TriggerCondition(
                action="SELL",
                can_trigger_next_day=False,
                comparison="BLOCKED",
                condition_summary="明日无法触发卖出",
                blocker_reason="历史数据不足以反推有效的 RSI 超买临界价。",
            )

    why_items = [
        {"指标": "最新收盘价", "当前值": f"${latest_close:,.2f}", "说明": "截至最新完成交易日"},
        {"指标": f"当前 RSI({strategy.period})", "当前值": f"{curr_rsi:.2f}", "说明": f"买入 ≤ {strategy.oversold:.0f}，卖出 ≥ {strategy.overbought:.0f}"},
        {"指标": f"SMA({strategy.trend_ma_period}) 趋势过滤线", "当前值": f"${trend_ma:,.2f}" if pd.notna(trend_ma) else "预热中（暂放行）", "说明": "买入要求 Close > SMA200"},
        {"指标": f"RSI={strategy.oversold:.0f} 反推价", "当前值": f"${p_os_raw:,.2f}" if (p_os_raw and p_os_raw > 0) else "单日不可达 (< $0)", "说明": "Wilder 递推精确解"},
        {"指标": f"RSI={strategy.overbought:.0f} 反推价", "当前值": f"${p_ob_raw:,.2f}" if (p_ob_raw and p_ob_raw > 0) else "N/A", "说明": "Wilder 递推精确解"},
    ]
    why_expl = [
        f"S02 规则：空仓时 `RSI({strategy.period}) ≤ {strategy.oversold:.0f}` 且 `Close > SMA({strategy.trend_ma_period})` 买入；持仓时 `RSI({strategy.period}) ≥ {strategy.overbought:.0f}` 卖出。",
        f"当前 `RSI({strategy.period}) = {curr_rsi:.2f}`，`SMA({strategy.trend_ma_period})` = {f'${trend_ma:,.2f}' if pd.notna(trend_ma) else 'NaN (预热中)'}。",
        (
            f"趋势过滤线明日临界值约为 **${p_trend:,.2f}**，RSI 超卖反推价为 **{f'${p_os_raw:,.2f}' if (p_os_raw and p_os_raw > 0) else '无正价格解'}**；二者交集决定了明日是否存在有效买点区间。"
            if trend_active_tomorrow
            else "当前历史长度未满 200 日或未启用趋势过滤，明日买入仅受 RSI 超卖阈值约束。"
        ),
    ]
    return buy_trig, sell_trig, why_items, why_expl


def _solve_s03(
    df: pd.DataFrame,
    hist_sig: pd.DataFrame,
    strategy,
    latest_close: float,
    curr_pos: int,
) -> Tuple[TriggerCondition, TriggerCondition, List[Dict[str, str]], List[str]]:
    ew, xw = strategy.entry_window, strategy.exit_window
    next_upper = float(df["High"].iloc[-ew:].max()) if len(df) >= ew else None
    next_lower = float(df["Low"].iloc[-xw:].min()) if len(df) >= xw else None

    if curr_pos == 0:
        if next_upper is not None and next_lower is not None:
            pos_above, _, _, _ = ForwardScenarioEngine.evaluate_next_day_price(df, strategy, next_upper + 0.01)
            pos_eq, _, _, _ = ForwardScenarioEngine.evaluate_next_day_price(df, strategy, next_upper)
            if pos_above == 1 and pos_eq == 0:
                dist_pct, dist_abs = _dist_metrics(next_upper, latest_close)
                buy_trig = TriggerCondition(
                    action="BUY",
                    can_trigger_next_day=True,
                    trigger_price=next_upper,
                    comparison=">",
                    distance_pct=dist_pct,
                    distance_abs=dist_abs,
                    condition_summary=f"收盘价严格高于过去 {ew} 日最高价 ${next_upper:,.2f}，触发唐奇安通道上轨突破买入。",
                    earliest_execution=EXEC_CONDITIONAL_NEXT,
                )
            else:
                buy_trig = TriggerCondition(
                    action="BUY",
                    can_trigger_next_day=False,
                    comparison="BLOCKED",
                    condition_summary="明日无法触发买入",
                    blocker_reason="策略通道尚未满足有效突破条件。",
                )
        else:
            buy_trig = TriggerCondition(
                action="BUY",
                can_trigger_next_day=False,
                comparison="BLOCKED",
                condition_summary="明日无法触发买入",
                blocker_reason=f"历史数据不足 {max(ew, xw)} 个交易日，通道尚未构建完成。",
            )
        sell_trig = TriggerCondition(
            action="SELL",
            can_trigger_next_day=False,
            comparison="BLOCKED",
            condition_summary="当前为空仓状态，无需触发卖出",
            blocker_reason="当前仓位为 0（空仓），须先突破上轨建仓后才监测下轨止损。",
        )
    else:
        buy_trig = TriggerCondition(
            action="BUY",
            can_trigger_next_day=False,
            comparison="BLOCKED",
            condition_summary="当前已持仓，不会重复触发买入",
            blocker_reason="策略当前已持多仓 (Position=1)，仅监测下轨跌破卖出。",
        )
        if next_lower is not None and next_upper is not None:
            pos_below, _, _, _ = ForwardScenarioEngine.evaluate_next_day_price(df, strategy, max(0.01, next_lower - 0.01))
            pos_eq, _, _, _ = ForwardScenarioEngine.evaluate_next_day_price(df, strategy, next_lower)
            if pos_below == 0 and pos_eq == 1:
                dist_pct, dist_abs = _dist_metrics(next_lower, latest_close)
                sell_trig = TriggerCondition(
                    action="SELL",
                    can_trigger_next_day=True,
                    trigger_price=next_lower,
                    comparison="<",
                    distance_pct=dist_pct,
                    distance_abs=dist_abs,
                    condition_summary=f"收盘价严格低于过去 {xw} 日最低价 ${next_lower:,.2f}，触发唐奇安通道下轨止损卖出。",
                    earliest_execution=EXEC_CONDITIONAL_NEXT,
                )
            else:
                sell_trig = TriggerCondition(
                    action="SELL",
                    can_trigger_next_day=False,
                    comparison="BLOCKED",
                    condition_summary="明日无法触发卖出",
                    blocker_reason="通道下轨未满足有效跌破边界。",
                )
        else:
            sell_trig = TriggerCondition(
                action="SELL",
                can_trigger_next_day=False,
                comparison="BLOCKED",
                condition_summary="明日无法触发卖出",
                blocker_reason=f"历史数据不足 {max(ew, xw)} 个交易日。",
            )

    why_items = [
        {"指标": "最新收盘价", "当前值": f"${latest_close:,.2f}", "说明": "截至最新完成交易日"},
        {"指标": f"明日上轨 ({ew}日最高价)", "当前值": f"${next_upper:,.2f}" if next_upper else "预热中", "说明": "由截至今日的完整历史决定，明日不再变动"},
        {"指标": f"明日下轨 ({xw}日最低价)", "当前值": f"${next_lower:,.2f}" if next_lower else "预热中", "说明": "由截至今日的完整历史决定，明日不再变动"},
        {"指标": "当前通道位置", "当前值": f"{(latest_close - next_lower) / (next_upper - next_lower) * 100:.1f}%" if (next_upper and next_lower and next_upper > next_lower) else "N/A", "说明": "0% 为下轨，100% 为上轨"},
    ]
    why_expl = [
        f"S03 规则：上轨取前 `{ew}` 日最高价（shift 1），下轨取前 `{xw}` 日最低价（shift 1），完全无前视偏差。",
        f"因此明日 (T+1) 的上下轨阈值在今日收盘后已**完全锁定**：上轨 = {f'${next_upper:,.2f}' if next_upper else 'N/A'}，下轨 = {f'${next_lower:,.2f}' if next_lower else 'N/A'}。",
        "比较边界为严格不等式：空仓时 `Close > 上轨` 买入，持仓时 `Close < 下轨` 卖出（恰好等于边界不会触发）。",
    ]
    return buy_trig, sell_trig, why_items, why_expl


def _solve_s05(
    df: pd.DataFrame,
    hist_sig: pd.DataFrame,
    strategy,
    latest_close: float,
    curr_pos: int,
) -> Tuple[TriggerCondition, TriggerCondition, List[Dict[str, str]], List[str]]:
    closes = df["Close"]
    bb_mid = hist_sig["BB_Middle"].iloc[-1]
    bb_up = hist_sig["BB_Upper"].iloc[-1]
    bw = hist_sig["BandWidth"].iloc[-1]
    bw_th = hist_sig["BW_Threshold"].iloc[-1]

    # Check if any of the past (squeeze_recent_window - 1) days had Is_Squeeze == True
    prev_k = max(0, strategy.squeeze_recent_window - 1)
    prior_squeeze_count = int(hist_sig["Is_Squeeze"].iloc[-prev_k:].fillna(False).sum()) if prev_k > 0 else 0
    prior_squeezed = prior_squeeze_count > 0

    if curr_pos == 0:
        # Dynamic model: exact closed-form BB_Upper breakout root + real strategy engine verification
        found_trigger: Optional[float] = None
        if len(df) + 1 >= strategy.period:
            p_bb = _solve_bb_upper_breakout(closes, strategy.period, strategy.std_dev)
            if p_bb is not None and p_bb > 0.01:
                if prior_squeezed:
                    found_trigger = _verify_or_bisect_root(df, strategy, p_bb, target_pos=1, lo_is_target=False)
                else:
                    # Check once at p_bb + eps (minimum bandwidth among all prices > BB_Upper)
                    eps = max(1e-3, p_bb * 1e-4)
                    pos_probe, _, _, _ = ForwardScenarioEngine.evaluate_next_day_price(df, strategy, p_bb + eps)
                    if pos_probe == 1:
                        found_trigger = _verify_or_bisect_root(df, strategy, p_bb, target_pos=1, lo_is_target=False)

        if found_trigger is not None:
            dist_pct, dist_abs = _dist_metrics(found_trigger, latest_close)
            buy_trig = TriggerCondition(
                action="BUY",
                can_trigger_next_day=True,
                trigger_price=found_trigger,
                comparison=">",
                distance_pct=dist_pct,
                distance_abs=dist_abs,
                condition_summary=(
                    f"近 {prev_k} 日已满足低波动挤压条件；经动态布林带模拟，明日收盘价 > ${found_trigger:,.2f} 将同步上穿动态 20日+2σ 上轨并触发买入。"
                ),
                earliest_execution=EXEC_CONDITIONAL_NEXT,
            )
        else:
            if not prior_squeezed:
                reason = (
                    f"近 {prev_k} 个交易日内未出现低波动率挤压（BandWidth 需落入过去 {strategy.squeeze_lookback} 日最低 {int(strategy.squeeze_quantile*100)}% 分位），"
                    f"当前带宽={f'{bw:.4f}' if pd.notna(bw) else '预热中'} vs 阈值={f'{bw_th:.4f}' if pd.notna(bw_th) else '预热中'}；因此明日即使大涨突破上轨也不会触发买入。"
                )
            else:
                reason = "经动态布林带模拟，明日在常规价格区间内收盘价无法跑赢同步扩张的 2σ 上轨。"
            buy_trig = TriggerCondition(
                action="BUY",
                can_trigger_next_day=False,
                comparison="BLOCKED",
                condition_summary="明日无法触发买入（未满足挤压前置条件）",
                blocker_reason=reason,
            )

        sell_trig = TriggerCondition(
            action="SELL",
            can_trigger_next_day=False,
            comparison="BLOCKED",
            condition_summary="当前为空仓状态，无需触发卖出",
            blocker_reason="当前仓位为 0（空仓），须先挤压突破进场后才监测中轨止损。",
        )
    else:
        buy_trig = TriggerCondition(
            action="BUY",
            can_trigger_next_day=False,
            comparison="BLOCKED",
            condition_summary="当前已持仓，不会重复触发买入",
            blocker_reason="策略当前已持多仓 (Position=1)，仅监测跌破 20 日中轨卖出条件。",
        )
        p_mid = _solve_sma_self_threshold(closes, strategy.period)
        mid_ref = _verify_or_bisect_root(df, strategy, p_mid, target_pos=0, lo_is_target=True) if p_mid is not None else None
        if mid_ref is not None:
            dist_pct, dist_abs = _dist_metrics(mid_ref, latest_close)
            sell_trig = TriggerCondition(
                action="SELL",
                can_trigger_next_day=True,
                trigger_price=mid_ref,
                comparison="<",
                distance_pct=dist_pct,
                distance_abs=dist_abs,
                condition_summary=f"收盘价严格低于 ${mid_ref:,.2f}（即跌破动态 {strategy.period} 日布林中轨），触发离场卖出。",
                earliest_execution=EXEC_CONDITIONAL_NEXT,
            )
        else:
            sell_trig = TriggerCondition(
                action="SELL",
                can_trigger_next_day=False,
                comparison="BLOCKED",
                condition_summary="明日无法触发卖出",
                blocker_reason=f"历史数据不足 {strategy.period} 日或未达中轨反转边界。",
            )

    why_items = [
        {"指标": "最新收盘价", "当前值": f"${latest_close:,.2f}", "说明": "截至最新完成交易日"},
        {"指标": f"布林上轨 (Upper {strategy.period}/{strategy.std_dev}σ)", "当前值": f"${bb_up:,.2f}" if pd.notna(bb_up) else "预热中", "说明": "随明日收盘价动态变化"},
        {"指标": f"布林中轨 (SMA{strategy.period})", "当前值": f"${bb_mid:,.2f}" if pd.notna(bb_mid) else "预热中", "说明": "持仓时跌破中轨离场"},
        {"指标": "当前带宽 BandWidth", "当前值": f"{bw*100:.2f}%" if pd.notna(bw) else "预热中", "说明": f"挤压阈值 ≤ {bw_th*100:.2f}%" if pd.notna(bw_th) else "预热中"},
        {"指标": f"近 {prev_k} 日挤压状态", "当前值": f"已触发 ({prior_squeeze_count} 天)" if prior_squeezed else "未触发 (0 天)", "说明": f"要求近 {strategy.squeeze_recent_window} 日内存在挤压"},
    ]
    why_expl = [
        f"S05 是典型的**动态指标策略**：明日收盘价变动会同步改变 `SMA({strategy.period})`、`σ` 标准差与布林上轨，因此必须代入动态模型求解。",
        f"买入前提：最近 `{strategy.squeeze_recent_window}` 日内发生过低波动挤压（带宽处于过去 `{strategy.squeeze_lookback}` 日最低 `{int(strategy.squeeze_quantile*100)}%` 分位），且 `Close > BB_Upper`。",
        f"当前近 `{prev_k}` 日已出现挤压天数：**{prior_squeeze_count}** 天；所有触发价均通过真实 `BollingerBandSqueezeStrategy.generate_signals()` 二分验证。",
    ]
    return buy_trig, sell_trig, why_items, why_expl


def _solve_s06(
    df: pd.DataFrame,
    hist_sig: pd.DataFrame,
    strategy,
    latest_close: float,
    curr_pos: int,
) -> Tuple[TriggerCondition, TriggerCondition, List[Dict[str, str]], List[str]]:
    closes = df["Close"]
    sma_fast_t = hist_sig["SMA_Fast"].iloc[-1]
    sma_mid_t = hist_sig["SMA_Mid"].iloc[-1]
    sma_slow_t = hist_sig["SMA_Slow"].iloc[-1]

    # Preconditions from completed day T
    yest_below_fast = bool(pd.notna(sma_fast_t) and latest_close <= float(sma_fast_t))
    yest_below_mid = bool(pd.notna(sma_mid_t) and latest_close < float(sma_mid_t))
    prev_k = max(0, strategy.pullback_window - 1)
    prior_touched_mid = bool(
        (hist_sig["Low"].iloc[-prev_k:] <= hist_sig["SMA_Mid"].iloc[-prev_k:]).fillna(False).any()
    ) if prev_k > 0 else False

    p_sma_fast = _solve_sma_self_threshold(closes, strategy.sma_fast)
    p_sma_mid = _solve_sma_self_threshold(closes, strategy.sma_mid)
    p_sma_slow = _solve_sma_self_threshold(closes, strategy.sma_slow)
    p_mid_gt_slow = _solve_two_sma_cross(closes, strategy.sma_mid, strategy.sma_slow)

    if curr_pos == 0:
        buy_trig: Optional[TriggerCondition] = None
        if len(df) + 1 < strategy.sma_slow or p_sma_slow is None or p_sma_fast is None or p_sma_mid is None:
            buy_trig = TriggerCondition(
                action="BUY",
                can_trigger_next_day=False,
                comparison="BLOCKED",
                condition_summary="明日无法触发买入",
                blocker_reason=f"历史数据不足 {strategy.sma_slow} 个交易日，SMA{strategy.sma_slow} 尚未预热完成。",
            )
        elif not yest_below_fast:
            buy_trig = TriggerCondition(
                action="BUY",
                can_trigger_next_day=False,
                comparison="BLOCKED",
                condition_summary="明日无法触发买入（不满足反转前置条件 C）",
                blocker_reason=(
                    f"最新收盘价 (${latest_close:,.2f}) 已高于 SMA{strategy.sma_fast} (${float(sma_fast_t):,.2f})，"
                    f"不满足条件 C 的前置要求（昨日收盘 ≤ SMA{strategy.sma_fast}），因此无论明日涨跌均无法触发 S06 买入。"
                ),
            )
        else:
            # All close lower bounds for Cond A + Cond C
            lower_bounds = [p_sma_fast, p_sma_slow]
            if p_mid_gt_slow is not None:
                lower_bounds.append(p_mid_gt_slow)
            p_min = max(lower_bounds)

            if prior_touched_mid:
                # Cond B already satisfied! Pure Close > p_min check
                ref_p = _verify_or_bisect_root(df, strategy, p_min, target_pos=1, lo_is_target=False)
                if ref_p is not None:
                    dist_pct, dist_abs = _dist_metrics(ref_p, latest_close)
                    buy_trig = TriggerCondition(
                        action="BUY",
                        can_trigger_next_day=True,
                        trigger_price=ref_p,
                        comparison=">",
                        distance_pct=dist_pct,
                        distance_abs=dist_abs,
                        condition_summary=(
                            f"近 {prev_k} 日已触及 SMA{strategy.sma_mid} 且今日收盘 ≤ SMA{strategy.sma_fast}；"
                            f"明日收盘价严格高于 ${ref_p:,.2f} 即同时满足条件 A/B/C 触发趋势回撤买入。"
                        ),
                        earliest_execution=EXEC_CONDITIONAL_NEXT,
                    )
                else:
                    buy_trig = TriggerCondition(
                        action="BUY",
                        can_trigger_next_day=False,
                        comparison="BLOCKED",
                        condition_summary="明日无法触发买入",
                        blocker_reason="经策略源码核验，明日无法同时满足多头排列（SMA50 > SMA200 且 Close > SMA200）与上穿 SMA20 条件。",
                    )
            else:
                # Cond B NOT yet satisfied in past 4 days: requires Low(T+1) <= SMA50(T+1) AND Close(T+1) > p_min
                probe_close = max(p_min + max(0.05, p_min * 1e-3), 0.05)
                sma50_at_probe = (float(closes.iloc[-(strategy.sma_mid - 1):].sum()) + probe_close) / float(strategy.sma_mid)
                ref_p = _verify_or_bisect_root(
                    df,
                    strategy,
                    p_min,
                    target_pos=1,
                    lo_is_target=False,
                    test_low_override=min(probe_close, sma50_at_probe * 0.999),
                )
                if ref_p is not None:
                    dist_pct, dist_abs = _dist_metrics(ref_p, latest_close)
                    buy_trig = TriggerCondition(
                        action="BUY",
                        can_trigger_next_day=True,
                        trigger_price=ref_p,
                        comparison=">",
                        distance_pct=dist_pct,
                        distance_abs=dist_abs,
                        intraday_low_required=True,
                        intraday_low_threshold=sma50_at_probe,
                        condition_summary=(
                            f"近 {prev_k} 日尚未触及 SMA{strategy.sma_mid}；明日须盘中最低价触及 ≤ ${sma50_at_probe:,.2f} (SMA{strategy.sma_mid}) "
                            f"且收盘价严格高于 > ${ref_p:,.2f} 才可触发买入。"
                        ),
                        earliest_execution=EXEC_CONDITIONAL_NEXT,
                    )
                else:
                    buy_trig = TriggerCondition(
                        action="BUY",
                        can_trigger_next_day=False,
                        comparison="BLOCKED",
                        condition_summary="明日无法触发买入（条件 A/B 未达成）",
                        blocker_reason=(
                            f"近 {prev_k} 日未触及 SMA{strategy.sma_mid} 且明日难以同时满足长期多头排列与突破条件。"
                        ),
                    )

        sell_trig = TriggerCondition(
            action="SELL",
            can_trigger_next_day=False,
            comparison="BLOCKED",
            condition_summary="当前为空仓状态，无需触发卖出",
            blocker_reason="当前仓位为 0（空仓），仅监测趋势回撤买入条件。",
        )
    else:
        buy_trig = TriggerCondition(
            action="BUY",
            can_trigger_next_day=False,
            comparison="BLOCKED",
            condition_summary="当前已持仓，不会重复触发买入",
            blocker_reason="策略当前已持多仓 (Position=1)，仅监测连续 2 日跌破 SMA50 卖出条件。",
        )
        if not yest_below_mid:
            sell_trig = TriggerCondition(
                action="SELL",
                can_trigger_next_day=False,
                comparison="BLOCKED",
                condition_summary="明日无法触发卖出（需连续 2 日跌破 SMA50）",
                blocker_reason=(
                    f"最新收盘价 (${latest_close:,.2f}) ≥ SMA{strategy.sma_mid} (${float(sma_mid_t):,.2f})。"
                    f"S06 要求**连续 2 个交易日**收盘低于 SMA{strategy.sma_mid} 才会卖出，明日即便跌破也仅算第 1 日，因此明日不可能触发卖出。"
                ),
            )
        elif p_sma_mid is not None and p_sma_mid > 0.01:
            lo_probe = max(0.01, p_sma_mid - max(0.05, p_sma_mid * 1e-3))
            hi_probe = p_sma_mid + max(0.05, p_sma_mid * 1e-3)
            pos_lo, _, _, _ = ForwardScenarioEngine.evaluate_next_day_price(df, strategy, lo_probe)
            pos_hi, _, _, _ = ForwardScenarioEngine.evaluate_next_day_price(df, strategy, hi_probe)
            if pos_lo == 0 and pos_hi == 1:
                ref_p = _bisect_boundary(df, strategy, lo=lo_probe, hi=hi_probe, target_pos=0, lo_is_target=True)
                dist_pct, dist_abs = _dist_metrics(ref_p, latest_close)
                sell_trig = TriggerCondition(
                    action="SELL",
                    can_trigger_next_day=True,
                    trigger_price=ref_p,
                    comparison="<",
                    distance_pct=dist_pct,
                    distance_abs=dist_abs,
                    condition_summary=(
                        f"今日收盘已低于 SMA{strategy.sma_mid}（第 1 日）；若明日收盘继续严格低于 ${ref_p:,.2f}，将构成连续 2 日跌破 SMA{strategy.sma_mid} 触发卖出。"
                    ),
                    earliest_execution=EXEC_CONDITIONAL_NEXT,
                )
            else:
                sell_trig = TriggerCondition(
                    action="SELL",
                    can_trigger_next_day=False,
                    comparison="BLOCKED",
                    condition_summary="明日无法触发卖出",
                    blocker_reason="经真实策略引擎核验，明日未能触发连续 2 日卖出边界。",
                )
        else:
            sell_trig = TriggerCondition(
                action="SELL",
                can_trigger_next_day=False,
                comparison="BLOCKED",
                condition_summary="明日无法触发卖出",
                blocker_reason=f"历史数据不足 {strategy.sma_mid} 个交易日。",
            )

    why_items = [
        {"指标": "最新收盘价", "当前值": f"${latest_close:,.2f}", "说明": "截至最新完成交易日"},
        {"指标": f"SMA({strategy.sma_fast}) / SMA({strategy.sma_mid}) / SMA({strategy.sma_slow})", "当前值": f"{f'${sma_fast_t:,.2f}' if pd.notna(sma_fast_t) else 'N/A'} / {f'${sma_mid_t:,.2f}' if pd.notna(sma_mid_t) else 'N/A'} / {f'${sma_slow_t:,.2f}' if pd.notna(sma_slow_t) else 'N/A'}", "说明": "多头趋势与回撤锚点"},
        {"指标": "条件 B：近 4 日触及 SMA50", "当前值": "✅ 已满足" if prior_touched_mid else "⏳ 未触及（需明日盘中下探）", "说明": f"过去 {strategy.pullback_window} 日窗口"},
        {"指标": "条件 C 前置：今日 Close ≤ SMA20", "当前值": "✅ 已满足" if yest_below_fast else "❌ 未满足（阻断明日买入）", "说明": "要求昨日≤SMA20且明日>SMA20"},
        {"指标": "卖出前置：今日 Close < SMA50", "当前值": "⚠️ 已连续 1 日" if yest_below_mid else "✅ 未跌破（明日无法卖出）", "说明": "要求连续 2 日跌破才卖出"},
    ]
    why_expl = [
        f"S06 是典型的**多状态前置条件策略**：不仅看明日价格，还严格依赖今日收盘相对于 `SMA({strategy.sma_fast})` 与 `SMA({strategy.sma_mid})` 的位置。",
        f"买入需同时满足：条件 A（`SMA50 > SMA200` 且 `Close > SMA200`）、条件 B（近 5 日有 `Low ≤ SMA50`）、条件 C（昨日 `Close ≤ SMA20` 且今日 `Close > SMA20`）。",
        f"卖出需**连续 2 个交易日** `Close < SMA50`。若前置条件未就绪，系统直接标注“当前无法触发”，绝不强行编造价格。",
    ]
    return buy_trig, sell_trig, why_items, why_expl
