"""
Interactive Altair Candlestick & Strategy Diagnosis Charts for Streamlit Web UI.
Provides crisp OHLC K-Line charts with accurate X/Y axes and strategy indicator overlays.
"""

from typing import Optional
import pandas as pd
import altair as alt


def _prepare_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes OHLCV columns and dates for Altair plotting."""
    plot_df = df.copy().reset_index()
    date_col = plot_df.columns[0]
    plot_df = plot_df.rename(columns={date_col: "Date"})
    plot_df["Date"] = pd.to_datetime(plot_df["Date"])
    plot_df["DateStr"] = plot_df["Date"].dt.strftime("%Y-%m-%d")
    plot_df["PctChange"] = plot_df["Close"].pct_change().fillna(0.0) * 100.0
    plot_df["Direction"] = plot_df.apply(
        lambda r: "上涨 (Close ≥ Open)" if r["Close"] >= r["Open"] else "下跌 (Close < Open)",
        axis=1,
    )
    return plot_df


def _x_encoding(plot_df: pd.DataFrame, show_title: bool = True) -> alt.X:
    """Selects crisp non-overlapping X-axis encoding for both short (3d/1w/1m) and multi-year spans."""
    n = len(plot_df)
    axis_title = "交易日期 (Trading Date)" if show_title else None

    if n <= 40:
        # Ordinal trading dates prevent weekend gaps & fractional hour ticks on 3d/1w/1m windows
        step = 1 if n <= 15 else 2
        values = plot_df["DateStr"].iloc[::step].tolist()
        return alt.X(
            "DateStr:O",
            sort=None,
            axis=alt.Axis(
                title=axis_title,
                values=values,
                labelAngle=-35,
                grid=True,
                gridOpacity=0.18,
            ),
        )
    date_fmt = "%Y-%m-%d" if n <= 160 else "%Y-%m"
    return alt.X(
        "Date:T",
        axis=alt.Axis(
            title=axis_title,
            format=date_fmt,
            grid=True,
            gridOpacity=0.18,
        ),
    )


def _bar_size(n: int) -> int:
    """Computes ideal candlestick body width based on number of trading days."""
    if n <= 5:
        return 36
    if n <= 25:
        return 16
    if n <= 130:
        return 5
    if n <= 300:
        return 3
    return 2


def build_period_candlestick_chart(
    ohlcv_df: pd.DataFrame,
    ticker: str,
    period_label: str = "",
) -> alt.VConcatChart:
    """
    Builds a 2-panel interactive OHLC Candlestick + Volume chart for the selected evaluation period,
    with complete, accurate X (Date) and Y (Price USD / Volume) axes.
    """
    plot_df = _prepare_ohlcv_df(ohlcv_df)
    n = len(plot_df)
    bar_w = _bar_size(n)

    x_top = _x_encoding(plot_df, show_title=False)
    x_bot = _x_encoding(plot_df, show_title=True)

    color_scale = alt.Scale(
        domain=["上涨 (Close ≥ Open)", "下跌 (Close < Open)"],
        range=["#10B981", "#EF4444"],
    )

    tooltips = [
        alt.Tooltip("DateStr:N", title="日期"),
        alt.Tooltip("Open:Q", title="开盘 (Open)", format="$.2f"),
        alt.Tooltip("High:Q", title="最高 (High)", format="$.2f"),
        alt.Tooltip("Low:Q", title="最低 (Low)", format="$.2f"),
        alt.Tooltip("Close:Q", title="收盘 (Close)", format="$.2f"),
        alt.Tooltip("PctChange:Q", title="日涨跌幅 (%)", format="+.2f"),
    ]
    if "Volume" in plot_df.columns:
        tooltips.append(alt.Tooltip("Volume:Q", title="成交量", format=",.0f"))

    y_min = float(plot_df["Low"].min()) * 0.985
    y_max = float(plot_df["High"].max()) * 1.015

    # High-Low wick
    wick = alt.Chart(plot_df).mark_rule(strokeWidth=1.4).encode(
        x=x_top,
        y=alt.Y("Low:Q", scale=alt.Scale(domain=[y_min, y_max]), axis=alt.Axis(title="股价 ($ USD)", format="$.2f")),
        y2="High:Q",
        color=alt.Color("Direction:N", scale=color_scale, legend=alt.Legend(title="K线方向", orient="top")),
        tooltip=tooltips,
    )

    # Open-Close candle body
    body = alt.Chart(plot_df).mark_bar(size=bar_w).encode(
        x=x_top,
        y=alt.Y("Open:Q", scale=alt.Scale(domain=[y_min, y_max])),
        y2="Close:Q",
        color=alt.Color("Direction:N", scale=color_scale, legend=None),
        tooltip=tooltips,
    )

    candle_panel = alt.layer(wick, body).properties(
        height=330,
        title=f"{ticker} K线走势图 ({period_label} | {plot_df['DateStr'].iloc[0]} 至 {plot_df['DateStr'].iloc[-1]}, 共 {n} 个交易日)",
    )

    if "Volume" in plot_df.columns and plot_df["Volume"].sum() > 0:
        vol_panel = alt.Chart(plot_df).mark_bar(size=max(2, bar_w)).encode(
            x=x_bot,
            y=alt.Y("Volume:Q", axis=alt.Axis(title="成交量 (股)", format="~s")),
            color=alt.Color("Direction:N", scale=color_scale, legend=None),
            tooltip=[alt.Tooltip("DateStr:N", title="日期"), alt.Tooltip("Volume:Q", title="成交量", format=",.0f")],
        ).properties(height=95)
        return alt.vconcat(candle_panel, vol_panel, spacing=8).resolve_scale(x="shared")

    return candle_panel.encode(x=x_bot)


def build_diagnosis_candlestick_chart(
    result,
    strategy_code: str,
    is_portfolio: bool = False,
    prices_df: Optional[pd.DataFrame] = None,
) -> alt.Chart:
    """
    Builds a strategy-specific diagnostic K-Line + Trigger Condition chart
    explaining visually why X trades (or 0 trades) occurred.
    """
    if is_portfolio:
        return _build_portfolio_diagnosis_chart(result, prices_df)
    return _build_single_diagnosis_chart(result, strategy_code)


def _build_single_diagnosis_chart(result, strategy_code: str) -> alt.Chart:
    """Single-stock K-Line with technical thresholds, blocked candidates, and trade signals."""
    df = _prepare_ohlcv_df(result.equity_df)
    code = strategy_code.upper()
    n = len(df)
    bar_w = _bar_size(n)
    x_enc = _x_encoding(df, show_title=True)
    x_no_title = _x_encoding(df, show_title=False)

    y_min = float(df["Low"].min()) * 0.97
    y_max = float(df["High"].max()) * 1.03

    color_scale = alt.Scale(
        domain=["上涨 (Close ≥ Open)", "下跌 (Close < Open)"],
        range=["#10B981", "#EF4444"],
    )

    # Base Candlestick layer (no extra legend clutter so indicator lines stand out)
    wick = alt.Chart(df).mark_rule(strokeWidth=1.2, opacity=0.75).encode(
        x=x_no_title if code in {"S02", "S05"} else x_enc,
        y=alt.Y("Low:Q", scale=alt.Scale(domain=[y_min, y_max]), axis=alt.Axis(title="股价 ($ USD)", format="$.2f")),
        y2="High:Q",
        color=alt.Color("Direction:N", scale=color_scale, legend=None),
        tooltip=[
            alt.Tooltip("DateStr:N", title="日期"),
            alt.Tooltip("Close:Q", title="收盘价", format="$.2f"),
            alt.Tooltip("Low:Q", title="最低价", format="$.2f"),
            alt.Tooltip("High:Q", title="最高价", format="$.2f"),
        ],
    )
    body = alt.Chart(df).mark_bar(size=bar_w, opacity=0.75).encode(
        x=x_no_title if code in {"S02", "S05"} else x_enc,
        y=alt.Y("Open:Q", scale=alt.Scale(domain=[y_min, y_max])),
        y2="Close:Q",
        color=alt.Color("Direction:N", scale=color_scale, legend=None),
    )
    layers = [wick, body]

    # Add indicator overlays by strategy
    if code == "S01" and "MA_Short" in df.columns and "MA_Long" in df.columns:
        ma_df = df.melt(
            id_vars=["Date", "DateStr"],
            value_vars=["MA_Short", "MA_Long"],
            var_name="均线指标",
            value_name="Value",
        ).dropna()
        ma_df["均线指标"] = ma_df["均线指标"].replace({"MA_Short": "SMA 20 (短期)", "MA_Long": "SMA 50 (长期)"})
        layers.append(
            alt.Chart(ma_df).mark_line(strokeWidth=2.0).encode(
                x=_x_encoding(df, show_title=True),
                y="Value:Q",
                color=alt.Color("均线指标:N", scale=alt.Scale(range=["#38BDF8", "#F59E0B"]), legend=alt.Legend(orient="top")),
            )
        )

    elif code == "S02" and "Trend_MA" in df.columns:
        trend_df = df[["Date", "DateStr", "Trend_MA"]].dropna()
        if not trend_df.empty:
            trend_df["指标"] = "200日牛熊过滤线 (SMA200)"
            layers.append(
                alt.Chart(trend_df).mark_line(strokeWidth=2.0, strokeDash=[5, 3]).encode(
                    x=x_no_title,
                    y="Trend_MA:Q",
                    color=alt.Color("指标:N", scale=alt.Scale(range=["#F59E0B"]), legend=alt.Legend(orient="top")),
                )
            )
        # Highlight days where RSI <= 30 (to visually prove blocked vs valid oversold signals!)
        oversold_df = df[df["RSI"] <= 30.0].copy()
        if not oversold_df.empty:
            oversold_df["状态"] = oversold_df.apply(
                lambda r: "有效超卖买点 (RSI≤30 & >SMA200)" if pd.isna(r["Trend_MA"]) or r["Close"] > r["Trend_MA"] else "被200日线拦截 (RSI≤30 但 <SMA200)",
                axis=1,
            )
            layers.append(
                alt.Chart(oversold_df).mark_point(size=160, filled=True, shape="diamond").encode(
                    x=x_no_title,
                    y="Low:Q",
                    color=alt.Color("状态:N", scale=alt.Scale(range=["#A855F7", "#22C55E"]), legend=alt.Legend(orient="top")),
                    tooltip=[
                        alt.Tooltip("DateStr:N", title="触发日期"),
                        alt.Tooltip("Close:Q", title="收盘价", format="$.2f"),
                        alt.Tooltip("Trend_MA:Q", title="SMA200", format="$.2f"),
                        alt.Tooltip("RSI:Q", title="RSI(14)", format=".2f"),
                        alt.Tooltip("状态:N", title="诊断结论"),
                    ],
                )
            )

    elif code == "S03" and "Upper_Channel" in df.columns:
        ch_df = df.melt(
            id_vars=["Date", "DateStr"],
            value_vars=["Upper_Channel", "Lower_Channel"],
            var_name="唐奇安通道",
            value_name="Value",
        ).dropna()
        ch_df["唐奇安通道"] = ch_df["唐奇安通道"].replace({"Upper_Channel": "20日上轨新高线", "Lower_Channel": "10日下轨止损线"})
        layers.append(
            alt.Chart(ch_df).mark_line(strokeWidth=1.8, strokeDash=[4, 3]).encode(
                x=x_enc,
                y="Value:Q",
                color=alt.Color("唐奇安通道:N", scale=alt.Scale(range=["#38BDF8", "#F43F5E"]), legend=alt.Legend(orient="top")),
            )
        )

    elif code == "S05" and "BB_Upper" in df.columns:
        bb_df = df.melt(
            id_vars=["Date", "DateStr"],
            value_vars=["BB_Upper", "BB_Middle", "BB_Lower"],
            var_name="布林带轨道",
            value_name="Value",
        ).dropna()
        bb_df["布林带轨道"] = bb_df["布林带轨道"].replace({"BB_Upper": "布林上轨", "BB_Middle": "20日均线 (中轨)", "BB_Lower": "布林下轨"})
        layers.append(
            alt.Chart(bb_df).mark_line(strokeWidth=1.6).encode(
                x=x_no_title,
                y="Value:Q",
                color=alt.Color("布林带轨道:N", scale=alt.Scale(range=["#38BDF8", "#F59E0B", "#38BDF8"]), legend=alt.Legend(orient="top")),
            )
        )

    elif code == "S06" and "SMA_Fast" in df.columns:
        pb_df = df.melt(
            id_vars=["Date", "DateStr"],
            value_vars=["SMA_Fast", "SMA_Mid", "SMA_Slow"],
            var_name="三均线结构",
            value_name="Value",
        ).dropna()
        pb_df["三均线结构"] = pb_df["三均线结构"].replace({"SMA_Fast": "SMA 20", "SMA_Mid": "SMA 50", "SMA_Slow": "SMA 200"})
        layers.append(
            alt.Chart(pb_df).mark_line(strokeWidth=1.8).encode(
                x=x_enc,
                y="Value:Q",
                color=alt.Color("三均线结构:N", scale=alt.Scale(range=["#38BDF8", "#F59E0B", "#A855F7"]), legend=alt.Legend(orient="top")),
            )
        )

    # Executed Buy/Sell Signal markers from closed trades
    if result.trades:
        trade_pts = []
        for t in result.trades:
            trade_pts.append({"Date": t.entry_date, "DateStr": t.entry_date.strftime("%Y-%m-%d"), "Price": t.entry_price, "Action": "买入信号 (Buy)"})
            trade_pts.append({"Date": t.exit_date, "DateStr": t.exit_date.strftime("%Y-%m-%d"), "Price": t.exit_price, "Action": "卖出信号 (Sell)"})
        pts_df = pd.DataFrame(trade_pts)
        layers.append(
            alt.Chart(pts_df).mark_point(size=140, filled=True).encode(
                x=x_no_title if code in {"S02", "S05"} else x_enc,
                y="Price:Q",
                shape=alt.Shape("Action:N", scale=alt.Scale(domain=["买入信号 (Buy)", "卖出信号 (Sell)"], range=["triangle-up", "triangle-down"])),
                color=alt.Color("Action:N", scale=alt.Scale(domain=["买入信号 (Buy)", "卖出信号 (Sell)"], range=["#22C55E", "#EF4444"]), legend=alt.Legend(orient="top")),
                tooltip=[alt.Tooltip("DateStr:N", title="日期"), alt.Tooltip("Action:N", title="动作"), alt.Tooltip("Price:Q", title="成交价格", format="$.2f")],
            )
        )

    top_chart = alt.layer(*layers).properties(
        height=310,
        title=f"{result.ticker} K线与策略信号诊断附图 ({result.strategy_name})",
    )

    # Extra Indicator Subplots for S02 (RSI) and S05 (BandWidth Squeeze)
    if code == "S02" and "RSI" in df.columns:
        rsi_line = alt.Chart(df).mark_line(color="#38BDF8", strokeWidth=1.8).encode(
            x=x_enc,
            y=alt.Y("RSI:Q", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(title="RSI(14)")),
            tooltip=[alt.Tooltip("DateStr:N", title="日期"), alt.Tooltip("RSI:Q", title="RSI(14)", format=".2f")],
        )
        os_rule = alt.Chart(pd.DataFrame({"y": [30.0]})).mark_rule(color="#22C55E", strokeDash=[4, 4], strokeWidth=1.5).encode(y="y:Q")
        ob_rule = alt.Chart(pd.DataFrame({"y": [70.0]})).mark_rule(color="#EF4444", strokeDash=[4, 4], strokeWidth=1.5).encode(y="y:Q")
        rsi_panel = alt.layer(rsi_line, os_rule, ob_rule).properties(
            height=130,
            title="RSI(14) 超卖(30) / 超买(70) 诊断轨道",
        )
        return alt.vconcat(top_chart, rsi_panel, spacing=8).resolve_scale(x="shared")

    if code == "S05" and "BandWidth" in df.columns and "BW_Threshold" in df.columns:
        bw_df = df[["Date", "DateStr", "BandWidth", "BW_Threshold"]].dropna()
        bw_melt = bw_df.melt(
            id_vars=["Date", "DateStr"],
            value_vars=["BandWidth", "BW_Threshold"],
            var_name="带宽诊断",
            value_name="Val",
        )
        bw_melt["带宽诊断"] = bw_melt["带宽诊断"].replace({"BandWidth": "布林带宽度 (BW)", "BW_Threshold": "挤压低波动阈值 (最低10%)"})
        bw_panel = alt.Chart(bw_melt).mark_line(strokeWidth=1.8).encode(
            x=x_enc,
            y=alt.Y("Val:Q", axis=alt.Axis(title="BandWidth")),
            color=alt.Color("带宽诊断:N", scale=alt.Scale(range=["#38BDF8", "#F59E0B"]), legend=alt.Legend(orient="top")),
        ).properties(height=130, title="布林带宽度挤压监测 (BW 低于橙线代表进入低波动蓄势区)")
        return alt.vconcat(top_chart, bw_panel, spacing=8).resolve_scale(x="shared")

    return top_chart


def _build_portfolio_diagnosis_chart(result, prices_df: Optional[pd.DataFrame]) -> alt.Chart:
    """Builds a normalized relative strength chart + Rebalance date rules for portfolio strategies."""
    eq_df = result.equity_df.copy().reset_index()
    date_col = eq_df.columns[0]
    eq_df = eq_df.rename(columns={date_col: "Date"})
    eq_df["Date"] = pd.to_datetime(eq_df["Date"])
    eq_df["DateStr"] = eq_df["Date"].dt.strftime("%Y-%m-%d")

    eq_df["Strategy_Norm"] = eq_df["Strategy_Equity"] / eq_df["Strategy_Equity"].iloc[0] * 100.0
    eq_df["Benchmark_Norm"] = eq_df["Benchmark_Equity"] / eq_df["Benchmark_Equity"].iloc[0] * 100.0

    melt_df = eq_df.melt(
        id_vars=["Date", "DateStr"],
        value_vars=["Strategy_Norm", "Benchmark_Norm"],
        var_name="曲线类型",
        value_name="归一化净值 (基期=100)",
    )
    melt_df["曲线类型"] = melt_df["曲线类型"].replace({
        "Strategy_Norm": f"组合策略 ({result.strategy_name})",
        "Benchmark_Norm": f"对比基准 ({getattr(result, 'benchmark_name', 'Benchmark')})",
    })

    x_enc = _x_encoding(eq_df, show_title=True)
    line_chart = alt.Chart(melt_df).mark_line(strokeWidth=2.2).encode(
        x=x_enc,
        y=alt.Y("归一化净值 (基期=100):Q", scale=alt.Scale(zero=False)),
        color=alt.Color("曲线类型:N", scale=alt.Scale(range=["#10B981", "#38BDF8"]), legend=alt.Legend(orient="top")),
        tooltip=[
            alt.Tooltip("DateStr:N", title="日期"),
            alt.Tooltip("曲线类型:N", title="曲线"),
            alt.Tooltip("归一化净值 (基期=100):Q", title="净值", format=".2f"),
        ],
    )

    layers = [line_chart]
    events = getattr(result, "rebalance_events", [])
    if events:
        ev_df = pd.DataFrame([
            {"Date": ev.date, "DateStr": ev.date.strftime("%Y-%m-%d"), "Label": "调仓日 (Rebalance)"}
            for ev in events
            if eq_df["Date"].iloc[0] <= ev.date <= eq_df["Date"].iloc[-1]
        ])
        if not ev_df.empty:
            layers.append(
                alt.Chart(ev_df).mark_rule(color="#F59E0B", strokeDash=[4, 4], strokeWidth=1.5).encode(
                    x="Date:T" if len(eq_df) > 40 else "DateStr:O",
                    tooltip=[alt.Tooltip("DateStr:N", title="调仓触发日期")],
                )
            )

    return alt.layer(*layers).properties(
        height=300,
        title=f"{result.strategy_name} 净值走势与调仓触发点图解 (黄色虚线=触发调仓/轮动日)",
    )
