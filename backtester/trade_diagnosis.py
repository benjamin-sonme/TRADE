"""
Trade Execution Diagnosis Module
Analyzes why a backtest executed X trades (or 0 trades) across any strategy (S01-S09)
and any evaluation timeframe, outputting human-readable reasoning and diagnostic statistics.
"""

from typing import Dict, Any, List
import pandas as pd


def _diagnose_single_stock(result, strategy_code: str) -> Dict[str, Any]:
    """Generates exact, data-backed diagnosis for single-stock strategies (S01, S02, S03, S05, S06)."""
    data = result.equity_df
    trades = result.trades
    code = strategy_code.upper()
    total_bars = len(data)
    closed_count = len(trades)

    pos_series = data["Position"] if "Position" in data.columns else pd.Series(0, index=data.index)
    holding_days = int((pos_series > 0).sum())
    currently_holding = bool(pos_series.iloc[-1] > 0) if total_bars > 0 else False

    # Identify entry/exit dates inside the window
    pos_diff = pos_series.diff().fillna(pos_series.iloc[0] if total_bars > 0 else 0)
    entry_dates = data.index[pos_diff > 0].tolist()
    exit_dates = data.index[pos_diff < 0].tolist()

    bullets: List[str] = []
    condition_rows: List[Dict[str, str]] = []

    if closed_count == 0 and holding_days == 0:
        headline = f"⚪ 本评估窗口（{total_bars} 个交易日）共发生 **0 笔交易**，策略全期保持空仓观望。"
    elif closed_count == 0 and holding_days > 0:
        if entry_dates:
            ed_str = entry_dates[-1].strftime("%Y-%m-%d")
            ep = data.loc[entry_dates[-1], "Open"]
            headline = (
                f"🔵 已触发买入但**尚未触发卖出**（持仓 {holding_days}/{total_bars} 个交易日，"
                f"进场日 {ed_str} @ ${ep:.2f}），因未完结平仓，故“已完成交易次数”统计为 **0 笔**。"
            )
        else:
            headline = (
                f"🔵 策略在窗口期首日起即保持多头持仓状态（共持仓 {holding_days}/{total_bars} 个交易日），"
                f"期间未触发离场卖出信号，因此“已完成平仓交易”统计为 **0 笔**。"
            )
    else:
        win_cnt = sum(1 for t in trades if t.pnl > 0)
        avg_days = sum(t.holding_days for t in trades) / closed_count
        headline = (
            f"🟢 本周期（{total_bars} 个交易日）共完结 **{closed_count} 笔**完整买卖交易"
            f"（其中盈利 {win_cnt} 笔，亏损 {closed_count - win_cnt} 笔，平均持仓 {avg_days:.1f} 天）"
            + ("，且期末仍持有 1 笔未平仓仓位。" if currently_holding else "。")
        )

    # --- Strategy-specific indicator bottleneck & condition inspection ---
    if code == "S01":
        # Dual MA Crossover (MA_Short vs MA_Long)
        bull_days = int((data["MA_Short"] > data["MA_Long"]).sum()) if "MA_Short" in data.columns else 0
        bear_days = total_bars - bull_days
        sig_diff = data["Signal"].diff().fillna(0) if "Signal" in data.columns else pd.Series(0, index=data.index)
        golden_crosses = int((sig_diff > 0).sum())
        death_crosses = int((sig_diff < 0).sum())

        last_short = data["MA_Short"].iloc[-1] if "MA_Short" in data.columns else float("nan")
        last_long = data["MA_Long"].iloc[-1] if "MA_Long" in data.columns else float("nan")

        condition_rows = [
            {"条件节点": "短均线高于长均线 (SMA20 > SMA50)", "发生次数/天数": f"{bull_days} 天 / {total_bars} 天", "说明": "满足持仓趋势条件的天数"},
            {"条件节点": "向上金叉买入信号 (Golden Cross)", "发生次数/天数": f"{golden_crosses} 次", "说明": "SMA20 从下方上穿 SMA50"},
            {"条件节点": "向下死叉卖出信号 (Death Cross)", "发生次数/天数": f"{death_crosses} 次", "说明": "SMA20 从上方跌破 SMA50"},
        ]
        if closed_count == 0 and holding_days == 0:
            bullets.append(
                f"📉 窗口期内 SMA20 共有 **{bear_days} 天**持续压制在 SMA50 下方（末日 SMA20=${last_short:.2f} vs SMA50=${last_long:.2f}），"
                f"从未形成有效向上金叉（0 次），策略按纪律全期空仓避险。"
            )
        elif closed_count == 0 and holding_days > 0:
            bullets.append(
                f"📈 窗口期内多头趋势极为连贯（共有 **{bull_days} 天** SMA20 > SMA50），仅发生 {death_crosses} 次死叉，策略持续锁定持仓享受主升浪，尚未平仓。"
            )
        else:
            bullets.append(
                f"⚡ 本阶段共出现 **{golden_crosses} 次金叉买入**信号与 **{death_crosses} 次死叉卖出**信号，"
                f"趋势切换触发了 {closed_count} 轮完整的低吸高抛或止损平仓。"
            )

    elif code == "S02":
        # RSI Mean Reversion (RSI <= 30 AND Close > Trend_MA)
        rsi = data["RSI"]
        rsi_min, rsi_max = float(rsi.min()), float(rsi.max())
        min_date = rsi.idxmin().strftime("%Y-%m-%d")
        oversold_mask = rsi <= 30.0
        oversold_days = int(oversold_mask.sum())
        overbought_days = int((rsi >= 70.0).sum())

        if "Trend_MA" in data.columns:
            trend_ok_mask = (data["Close"] > data["Trend_MA"]) | data["Trend_MA"].isna()
            trend_ok_days = int(trend_ok_mask.sum())
            both_mask = oversold_mask & trend_ok_mask
            both_days = int(both_mask.sum())
        else:
            trend_ok_days = total_bars
            both_days = oversold_days

        condition_rows = [
            {"条件节点": "RSI 超卖区间 (RSI ≤ 30)", "发生次数/天数": f"{oversold_days} 天 (最低 {rsi_min:.1f} @ {min_date})", "说明": "触发超卖买点的必要条件"},
            {"条件节点": "200日均线多头趋势过滤 (Close > SMA200)", "发生次数/天数": f"{trend_ok_days} 天 / {total_bars} 天", "说明": "防止在下跌熊市中盲目抄底"},
            {"条件节点": "同时满足买入条件 (RSI≤30 且 站上200日线)", "发生次数/天数": f"{both_days} 天", "说明": "真正允许开仓买入的信号天数"},
            {"条件节点": "RSI 超买获利区间 (RSI ≥ 70)", "发生次数/天数": f"{overbought_days} 天 (最高 {rsi_max:.1f})", "说明": "触发卖出止盈的信号天数"},
        ]

        if len(entry_dates) == 0 and holding_days == 0:
            if oversold_days == 0:
                bullets.append(
                    f"🛡️ **为何 0 笔交易？** 在这 {total_bars} 个交易日内，该股票走势较为坚挺，14 日 RSI 最低仅跌至 **{rsi_min:.2f}**（发生在 {min_date}），"
                    f"始终**未能触及 30.0 的极端超卖买点门槛**，因此模型严格按“不见兔子不撒鹰”纪律保持空仓。"
                )
            elif both_days == 0:
                sample_date = data.index[oversold_mask][0]
                s_close = float(data.loc[sample_date, "Close"])
                s_rsi = float(data.loc[sample_date, "RSI"])
                s_ma200 = float(data.loc[sample_date, "Trend_MA"])
                bullets.append(
                    f"🚧 **为何 0 笔交易？被 200 日牛熊线安全过滤拦截！** 周期内共出现过 **{oversold_days} 天** RSI ≤ 30 的超卖现象"
                    f"（如 `{sample_date.strftime('%Y-%m-%d')}` 当日 RSI 跌至 **{s_rsi:.2f}**，收盘价 **${s_close:.2f}**），"
                    f"但该日收盘价**低于 200 日牛熊分界线 (${s_ma200:.2f})**！为避免“下跌趋势接飞刀”，趋势过滤机制主动拦截了买单，这是保护本金的严谨风控设计。"
                )
        else:
            bullets.append(
                f"🎯 周期内共有 **{both_days} 天**同时满足“RSI ≤ 30 超卖”与“站稳 200 日线”，共执行了 **{len(entry_dates)} 次抄底买入**与 **{len(exit_dates)} 次超买/离场卖出**。"
            )
            if oversold_days > both_days:
                bullets.append(
                    f"🚧 此外另有 {oversold_days - both_days} 天虽 RSI 低于 30，但因当时跌破了 200 日均线而被趋势过滤网成功拦截。"
                )

    elif code == "S03":
        # Donchian Channel Breakout
        upper_break = int((data["Close"] > data["Upper_Channel"]).sum()) if "Upper_Channel" in data.columns else 0
        lower_break = int((data["Close"] < data["Lower_Channel"]).sum()) if "Lower_Channel" in data.columns else 0
        condition_rows = [
            {"条件节点": "收盘价突破过去 20 日上轨新高", "发生次数/天数": f"{upper_break} 天", "说明": "趋势向上突破买入触发条件"},
            {"条件节点": "收盘价跌破过去 10 日下轨新低", "发生次数/天数": f"{lower_break} 天", "说明": "趋势破坏止损/止盈条件"},
        ]
        if len(entry_dates) == 0 and holding_days == 0:
            bullets.append(
                f"📦 评估窗口内股价始终在唐奇安 20 日上轨与 10 日下轨箱体内部宽幅震荡（向上突破次数为 **{upper_break} 次**），未出现突破确认信号，故交易次数为 0。"
            )
        else:
            bullets.append(
                f"🚀 股价共录得 **{upper_break} 天**收在 20 日新高之上、**{lower_break} 天**跌破 10 日最低点，驱动模型完成了 {closed_count} 笔趋势突破交易。"
            )

    elif code == "S05":
        # Bollinger Band Squeeze
        squeeze_days = int(data["Is_Squeeze"].sum()) if "Is_Squeeze" in data.columns else 0
        recent_sq_days = int(data["Recent_Squeeze"].sum()) if "Recent_Squeeze" in data.columns else 0
        upper_break = int((data["Close"] > data["BB_Upper"]).sum()) if "BB_Upper" in data.columns else 0
        both_days = int(((data.get("Recent_Squeeze", 0) == 1) & (data["Close"] > data["BB_Upper"])).sum()) if "BB_Upper" in data.columns else 0

        condition_rows = [
            {"条件节点": "布林带宽落入半年最低 10% 挤压区", "发生次数/天数": f"{squeeze_days} 天", "说明": "代表极致低波动盘整蓄势"},
            {"条件节点": "股价向上放量突破布林带上轨", "发生次数/天数": f"{upper_break} 天", "说明": "向上方向性突破信号"},
            {"条件节点": "挤压后 5 日内成功向上突破上轨", "发生次数/天数": f"{both_days} 天", "说明": "严格策略触发买点"},
        ]
        if len(entry_dates) == 0 and holding_days == 0:
            if squeeze_days == 0:
                bullets.append(
                    f"🌊 **为何 0 笔交易？** 本周期内该股波动率始终较高，布林带宽度 (BandWidth) **从未收窄至过去半年最低 10% 挤压区间 (0 天)**，不符合“先蓄势盘整、再爆发突破”的模型前提。"
                )
            else:
                bullets.append(
                    f"⚡ **为何 0 笔交易？** 周期内曾有 **{squeeze_days} 天**进入低波动挤压区，但挤压后 5 日内**未能出现收盘突破上轨**的多头爆发确认（或向下突破），因此未触发开仓。"
                )
        else:
            bullets.append(
                f"💥 共出现 **{squeeze_days} 天**低波动盘整挤压期，并有 **{both_days} 次**成功捕捉到压缩后的向上爆发突破。"
            )

    elif code == "S06":
        # Trend Pullback
        cond_a = (data["SMA_Mid"] > data["SMA_Slow"]) & (data["Close"] > data["SMA_Slow"])
        cond_b = data["Low"].rolling(5).min() <= data["SMA_Mid"]
        cond_c = (data["Close"].shift(1) <= data["SMA_Fast"]) & (data["Close"] > data["SMA_Fast"])
        days_a = int(cond_a.sum())
        days_b = int(cond_b.sum())
        days_c = int(cond_c.sum())
        days_all = int((cond_a & cond_b & cond_c).sum())

        condition_rows = [
            {"条件节点": "条件 A: 长期大牛趋势 (SMA50>SMA200 且 股价>SMA200)", "发生次数/天数": f"{days_a} 天", "说明": "确认处于长期上升大势"},
            {"条件节点": "条件 B: 5日内低点回调触及 SMA50 支撑线", "发生次数/天数": f"{days_b} 天", "说明": "等待合理的健康回撤深度"},
            {"条件节点": "条件 C: 收盘价重新收复站上 SMA20", "发生次数/天数": f"{days_c} 天", "说明": "确认回调结束并重新转强"},
            {"条件节点": "A + B + C 严选共振共振触发买入", "发生次数/天数": f"{days_all} 天", "说明": "全部三条件同时满足"},
        ]
        if len(entry_dates) == 0 and holding_days == 0:
            if days_a == 0:
                bullets.append("📉 条件 A（SMA50 > SMA200 且股价站上 200 日线）满足 **0 天**，标的尚未处于长牛通道，模型拒绝逆势买入。")
            elif days_b == 0:
                bullets.append(
                    f"🚀 **为何 0 笔交易？强牛不回头！** 长牛条件 A 满足了 **{days_a} 天**，但股价过于强势，期间从未深幅回撤触碰过 50 日中期均线（条件 B = 0 天），因此未出现舒适的回踩买点。"
                )
            else:
                bullets.append(
                    f"⏳ 虽有 {days_a} 天长牛和 {days_b} 天触及 50 日线，但三项严苛条件未能与“收盘站回 20 日均线 (条件 C)”精准重叠（共振天数 = 0），故模型保持谨慎。"
                )
        else:
            bullets.append(
                f"🎯 模型共精准捕获了 **{days_all} 次**“长牛确认 + 回踩50日线 + 收复20日线”的三维共振买点，完成 {closed_count} 笔波段收益闭环。"
            )

    # Build readable closed trades table
    events_records = []
    for idx, t in enumerate(trades, 1):
        events_records.append({
            "序号": idx,
            "买入日期": t.entry_date.strftime("%Y-%m-%d"),
            "买入价格 ($)": f"${t.entry_price:.2f}",
            "卖出日期": t.exit_date.strftime("%Y-%m-%d"),
            "卖出价格 ($)": f"${t.exit_price:.2f}",
            "持仓天数": t.holding_days,
            "单笔收益率": f"{t.return_pct * 100:+.2f}%",
        })

    return {
        "headline": headline,
        "bullets": bullets,
        "condition_rows": condition_rows,
        "events_df": pd.DataFrame(events_records),
    }


def _diagnose_portfolio(result, strategy_code: str) -> Dict[str, Any]:
    """Generates diagnosis for multi-asset portfolio strategies (S04, S07, S08, S09)."""
    events = getattr(result, "rebalance_events", [])
    total_bars = len(result.equity_df)
    rebal_cnt = len(events)
    code = strategy_code.upper()

    bullets: List[str] = []
    condition_rows: List[Dict[str, str]] = []

    if rebal_cnt == 0:
        headline = f"⚪ 本评估窗口（{total_bars} 个交易日）内发生 **0 次调仓/再平衡**，组合沿用期初配置持有。"
    elif rebal_cnt == 1:
        headline = f"🔵 本周期（{total_bars} 个交易日）仅在期初完成 **1 次初始建仓配置**，后续仓位在安全带内未再触发换手。"
    else:
        headline = f"🟢 本周期（{total_bars} 个交易日）共执行了 **{rebal_cnt} 次动态调仓与轮动**，及时完成了强弱切换或仓位再平衡。"

    if code == "S04":
        condition_rows = [
            {"规则维度": "轮动频率", "配置标准": "每月最后一个交易日", "实际执行": f"触发 {rebal_cnt} 次月度动量排序切换"},
            {"规则维度": "动量因子公式", "配置标准": "12-1 累计动量 > 0 排名前 4", "实际执行": "每只配 20%，自动保留 ≥20% 现金防尾盘风险"},
        ]
        if rebal_cnt <= 1:
            bullets.append(
                f"⏱️ 所选时间窗口（{total_bars} 个交易日）较短，仅包含期初建仓或未跨越多个月末结算日，因此没有产生高频月度换手调仓。"
            )
        else:
            bullets.append(
                f"🔄 系统在每个月末精确计算候选股票池的 12-1 动量排名，剔除走弱标的并切换至最强 4 只龙头（共调仓 {rebal_cnt} 次）。"
            )

    elif code == "S07":
        condition_rows = [
            {"规则维度": "检查频率与容忍带", "配置标准": "月末检查 / ±25% 相对偏离容忍带", "实际执行": f"共触发 {rebal_cnt} 次超限再平衡操作"},
            {"规则维度": "现金风控底线", "配置标准": "目标 20% 现金缓冲", "实际执行": "超涨止盈减仓 / 低配按规则补仓"},
        ]
        if rebal_cnt <= 1:
            bullets.append(
                f"⚖️ 期间资产波动温和或时间跨度较短，各资产实际占比始终稳居预设的 **±25% 容忍区间**内，无需无意义摩擦交易，维持 0 次额外调仓。"
            )
        else:
            bullets.append(
                f"⚖️ 在 {total_bars} 个交易日中共有 **{rebal_cnt} 个调仓事件** triggered，在某资产大涨超标时严格落袋为安、超跌时低位补足权重。"
            )

    elif code == "S08":
        condition_rows = [
            {"规则维度": "季频财报重筛选", "配置标准": "每 ~63 交易日一次", "实际执行": f"共完成 {rebal_cnt} 次财报质量与 FCF Yield 重排"},
            {"规则维度": "质量防雷门槛", "配置标准": "ROIC>10%, FCF>0, 净负债/EBITDA<3", "实际执行": "优中选优持有前 5，保留 25% 现金"},
        ]
        bullets.append(
            f"🏛️ 质量与价值多因子属于中低频基本面配置策略（按季度重排），本窗口共经历了 {rebal_cnt} 次季频选股评估与动态权重调整。"
        )

    elif code == "S09":
        condition_rows = [
            {"规则维度": "双重动量比较 (SPY vs VEU vs BIL)", "配置标准": "月末检查 12 个月总回报", "实际执行": f"共执行 {rebal_cnt} 次宏观大类资产检测/切换"},
            {"规则维度": "绝对动量防御开关", "配置标准": "胜出股指必须跑赢 BIL 短期国债", "实际执行": "跑不赢国债时 100% 切入 BIL 避险"},
        ]
        bullets.append(
            f"🌍 双重动量策略在 {total_bars} 个交易日内共进行 {rebal_cnt} 次宏观切换与再平衡，动态在美股 (SPY)、国际股市 (VEU) 与国债现金 (BIL) 间防御切换。"
        )

    events_records = []
    for idx, ev in enumerate(events[:25], 1):
        top_allocs = sorted(
            [(k, v) for k, v in ev.actual_weights.items() if k != "CASH" and v > 0.005],
            key=lambda x: x[1],
            reverse=True,
        )
        alloc_str = ", ".join(f"{k}:{v*100:.1f}%" for k, v in top_allocs[:4])
        cash_w = ev.actual_weights.get("CASH", 0.0) * 100
        events_records.append({
            "序号": idx,
            "调仓日期": ev.date.strftime("%Y-%m-%d"),
            "组合总净值 ($)": f"${ev.portfolio_equity:,.2f}",
            "换手率": f"{ev.turnover*100:.1f}%",
            "核心持仓分布": alloc_str or "100% 现金防守",
            "现金比例": f"{cash_w:.1f}%",
            "备注": ev.notes or "定期动态调整",
        })

    return {
        "headline": headline,
        "bullets": bullets,
        "condition_rows": condition_rows,
        "events_df": pd.DataFrame(events_records),
    }


def diagnose_backtest_trades(result, strategy_code: str, is_portfolio: bool = False) -> Dict[str, Any]:
    """Unified entry point returning structured diagnosis and trade/rebalance details."""
    if is_portfolio:
        return _diagnose_portfolio(result, strategy_code)
    return _diagnose_single_stock(result, strategy_code)
