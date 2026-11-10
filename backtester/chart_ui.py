"""
Interactive Apache ECharts Candlestick & Strategy Diagnosis Charts for Streamlit Web UI.
Provides crisp OHLC K-Line charts with trackpad pinch-to-zoom, horizontal drag/slider,
accurate X/Y axes, and strict container-width containment inside Streamlit bordered cards.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


def _round_cols(df: pd.DataFrame, cols: List[str], decimals: int = 2) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(decimals)
    return out


def _base_zoom_and_toolbox(x_indices: List[int]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Configures interactive zoom (trackpad two-finger pinch, drag pan, bottom slider,
    and top-left box-zoom / reset toolbox) across all linked X axes.
    """
    data_zoom = [
        {
            "type": "inside",
            "xAxisIndex": x_indices,
            "zoomOnMouseWheel": True,
            "moveOnMouseMove": True,
            "moveOnMouseWheel": False,
            "minSpan": 2,
        },
        {
            "type": "slider",
            "xAxisIndex": x_indices,
            "show": True,
            "bottom": 8,
            "height": 24,
            "start": 0,
            "end": 100,
            "borderColor": "#334155",
            "fillerColor": "rgba(56, 189, 248, 0.20)",
            "handleStyle": {"color": "#38BDF8"},
        },
    ]
    toolbox = {
        "left": 8,
        "top": 0,
        "feature": {
            "dataZoom": {
                "yAxisIndex": "none",
                "title": {"zoom": "框选放大", "back": "上一步缩放"},
            },
            "restore": {"title": "重置视图"},
        },
    }
    return data_zoom, toolbox


def build_period_candlestick_chart(
    ohlcv_df: pd.DataFrame,
    ticker: str,
    period_label: str = "",
) -> Tuple[Dict[str, Any], int, str]:
    """
    Builds an Apache ECharts 2-grid (Price Candlestick + Volume) option dict
    for the top Hero period preview, supporting trackpad pinch zoom & bottom slider.
    Returns: (echarts_option, height_px, alt_text)
    """
    src = ohlcv_df.copy().reset_index()
    date_col = src.columns[0]
    src = src.rename(columns={date_col: "Date"})
    src["Date"] = pd.to_datetime(src["Date"]).dt.strftime("%Y-%m-%d")

    has_vol = "Volume" in src.columns and float(src["Volume"].fillna(0).max()) > 0
    if not has_vol:
        src["Volume"] = 0

    src["VolUp"] = np.where(src["Close"] >= src["Open"], src["Volume"].fillna(0), 0)
    src["VolDown"] = np.where(src["Close"] < src["Open"], src["Volume"].fillna(0), 0)
    src = _round_cols(src, ["Open", "High", "Low", "Close"], 2)

    keep_cols = ["Date", "Open", "Close", "Low", "High", "VolUp", "VolDown"]
    dataset_df = src[keep_cols].copy()

    x_indices = [0, 1] if has_vol else [0]
    data_zoom, toolbox = _base_zoom_and_toolbox(x_indices)

    if has_vol:
        grids = [
            {"left": 18, "right": 18, "top": 64, "height": "53%", "containLabel": True},
            {"left": 18, "right": 18, "top": "70%", "height": "14%", "containLabel": True},
        ]
        x_axes = [
            {
                "type": "category",
                "boundaryGap": True,
                "axisLine": {"onZero": False},
                "axisLabel": {"show": False},
                "axisTick": {"show": False},
            },
            {
                "type": "category",
                "gridIndex": 1,
                "boundaryGap": True,
                "axisLine": {"onZero": False},
                "axisLabel": {"show": True},
            },
        ]
        y_axes = [
            {
                "scale": True,
                "axisLabel": {"formatter": "${value}"},
                "splitArea": {
                    "show": True,
                    "areaStyle": {"color": ["rgba(255,255,255,0.01)", "rgba(255,255,255,0.025)"]},
                },
            },
            {
                "scale": True,
                "gridIndex": 1,
                "splitNumber": 2,
                "axisLabel": {"show": True},
            },
        ]
        series = [
            {
                "name": f"{ticker} K线",
                "type": "candlestick",
                "encode": {
                    "x": "Date",
                    "y": ["Open", "Close", "Low", "High"],
                    "tooltip": ["Open", "High", "Low", "Close"],
                },
                "itemStyle": {
                    "color": "#10B981",
                    "color0": "#EF4444",
                    "borderColor": "#10B981",
                    "borderColor0": "#EF4444",
                },
                "tooltip": {"valueFormatter": "dollar"},
            },
            {
                "name": "上涨成交量",
                "type": "bar",
                "xAxisIndex": 1,
                "yAxisIndex": 1,
                "stack": "vol",
                "encode": {"x": "Date", "y": "VolUp"},
                "itemStyle": {"color": "rgba(16, 185, 129, 0.7)"},
                "tooltip": {"valueFormatter": "compact"},
            },
            {
                "name": "下跌成交量",
                "type": "bar",
                "xAxisIndex": 1,
                "yAxisIndex": 1,
                "stack": "vol",
                "encode": {"x": "Date", "y": "VolDown"},
                "itemStyle": {"color": "rgba(239, 68, 68, 0.7)"},
                "tooltip": {"valueFormatter": "compact"},
            },
        ]
        height_px = 470
    else:
        grids = [{"left": 18, "right": 18, "top": 64, "bottom": 66, "containLabel": True}]
        x_axes = [{"type": "category", "boundaryGap": True, "axisLine": {"onZero": False}}]
        y_axes = [{"scale": True, "axisLabel": {"formatter": "${value}"}}]
        series = [
            {
                "name": f"{ticker} K线",
                "type": "candlestick",
                "encode": {
                    "x": "Date",
                    "y": ["Open", "Close", "Low", "High"],
                    "tooltip": ["Open", "High", "Low", "Close"],
                },
                "itemStyle": {
                    "color": "#10B981",
                    "color0": "#EF4444",
                    "borderColor": "#10B981",
                    "borderColor0": "#EF4444",
                },
                "tooltip": {"valueFormatter": "dollar"},
            }
        ]
        height_px = 400

    option = {
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {"type": "cross"},
        },
        "legend": {"top": 30, "left": "center"},
        "toolbox": toolbox,
        "axisPointer": {"link": [{"xAxisIndex": "all"}]},
        "grid": grids,
        "xAxis": x_axes,
        "yAxis": y_axes,
        "dataZoom": data_zoom,
        "dataset": {"source": dataset_df},
        "series": series,
    }
    alt_text = f"{ticker} OHLC Candlestick chart for {period_label}"
    return option, height_px, alt_text


def build_diagnosis_candlestick_charts(
    result,
    strategy_code: str,
    is_portfolio: bool = False,
) -> List[Tuple[str, Dict[str, Any], int, str]]:
    """
    Returns a list of (section_subtitle, echarts_option, height_px, alt_text) tuples
    visually explaining why X trades (or 0 trades) occurred, with full pinch zoom & slider support.
    """
    if is_portfolio:
        opt, h, alt_txt = _build_portfolio_diagnosis_chart(result)
        return [("组合策略 vs 基准归一化走势（虚线为触发调仓/轮动日）", opt, h, alt_txt)]
    return _build_single_diagnosis_charts(result, strategy_code)


def _build_single_diagnosis_charts(
    result,
    strategy_code: str,
) -> List[Tuple[str, Dict[str, Any], int, str]]:
    """Builds interactive ECharts candlestick + indicator overlay + buy/sell/blocked triggers."""
    src = result.equity_df.copy().reset_index()
    date_col = src.columns[0]
    src = src.rename(columns={date_col: "Date"})
    src["Date"] = pd.to_datetime(src["Date"]).dt.strftime("%Y-%m-%d")

    code = strategy_code.upper()
    has_sub_panel = (code == "S02" and "RSI" in src.columns) or (
        code == "S05" and "BandWidth" in src.columns and "BW_Threshold" in src.columns
    )

    # Buy & Sell scatter columns
    src["BuySignal"] = np.nan
    src["SellSignal"] = np.nan
    if result.trades:
        buy_map = {t.entry_date.strftime("%Y-%m-%d"): round(float(t.entry_price), 2) for t in result.trades}
        sell_map = {t.exit_date.strftime("%Y-%m-%d"): round(float(t.exit_price), 2) for t in result.trades}
        src["BuySignal"] = src["Date"].map(buy_map)
        src["SellSignal"] = src["Date"].map(sell_map)

    round_targets = ["Open", "High", "Low", "Close"]
    series: List[Dict[str, Any]] = [
        {
            "name": f"{result.ticker} K线",
            "type": "candlestick",
            "encode": {
                "x": "Date",
                "y": ["Open", "Close", "Low", "High"],
                "tooltip": ["Open", "High", "Low", "Close"],
            },
            "itemStyle": {
                "color": "#10B981",
                "color0": "#EF4444",
                "borderColor": "#10B981",
                "borderColor0": "#EF4444",
            },
            "tooltip": {"valueFormatter": "dollar"},
        }
    ]
    keep_cols = ["Date", "Open", "Close", "Low", "High", "BuySignal", "SellSignal"]

    # Add indicator series cleanly by strategy
    if code == "S01" and "MA_Short" in src.columns and "MA_Long" in src.columns:
        keep_cols.extend(["MA_Short", "MA_Long"])
        round_targets.extend(["MA_Short", "MA_Long"])
        series.extend([
            {
                "name": "SMA 20 (短期)",
                "type": "line",
                "showSymbol": False,
                "smooth": False,
                "encode": {"x": "Date", "y": "MA_Short"},
                "lineStyle": {"width": 2, "color": "#38BDF8"},
                "itemStyle": {"color": "#38BDF8"},
                "tooltip": {"valueFormatter": "dollar"},
            },
            {
                "name": "SMA 50 (长期)",
                "type": "line",
                "showSymbol": False,
                "smooth": False,
                "encode": {"x": "Date", "y": "MA_Long"},
                "lineStyle": {"width": 2, "color": "#F59E0B"},
                "itemStyle": {"color": "#F59E0B"},
                "tooltip": {"valueFormatter": "dollar"},
            },
        ])

    elif code == "S02" and "Trend_MA" in src.columns:
        keep_cols.extend(["Trend_MA", "RSI", "BlockedOversold"])
        round_targets.extend(["Trend_MA", "RSI", "BlockedOversold"])
        blocked_mask = (src["RSI"] <= 30.0) & (~src["Trend_MA"].isna()) & (src["Close"] <= src["Trend_MA"])
        src["BlockedOversold"] = np.where(blocked_mask, src["Low"] * 0.985, np.nan)
        series.extend([
            {
                "name": "200日牛熊过滤线",
                "type": "line",
                "showSymbol": False,
                "encode": {"x": "Date", "y": "Trend_MA"},
                "lineStyle": {"width": 2, "type": "dashed", "color": "#F59E0B"},
                "itemStyle": {"color": "#F59E0B"},
                "tooltip": {"valueFormatter": "dollar"},
            },
            {
                "name": "被200日线拦截的超卖点",
                "type": "scatter",
                "symbol": "diamond",
                "symbolSize": 14,
                "encode": {"x": "Date", "y": "BlockedOversold"},
                "itemStyle": {"color": "#A855F7"},
                "tooltip": {"valueFormatter": "dollar"},
            },
        ])

    elif code == "S03" and "Upper_Channel" in src.columns:
        keep_cols.extend(["Upper_Channel", "Lower_Channel"])
        round_targets.extend(["Upper_Channel", "Lower_Channel"])
        series.extend([
            {
                "name": "20日上轨新高线",
                "type": "line",
                "showSymbol": False,
                "encode": {"x": "Date", "y": "Upper_Channel"},
                "lineStyle": {"width": 1.8, "type": "dashed", "color": "#38BDF8"},
                "itemStyle": {"color": "#38BDF8"},
                "tooltip": {"valueFormatter": "dollar"},
            },
            {
                "name": "10日下轨止损线",
                "type": "line",
                "showSymbol": False,
                "encode": {"x": "Date", "y": "Lower_Channel"},
                "lineStyle": {"width": 1.8, "type": "dashed", "color": "#F43F5E"},
                "itemStyle": {"color": "#F43F5E"},
                "tooltip": {"valueFormatter": "dollar"},
            },
        ])

    elif code == "S05" and "BB_Upper" in src.columns:
        keep_cols.extend(["BB_Upper", "BB_Middle", "BB_Lower", "BandWidth", "BW_Threshold"])
        round_targets.extend(["BB_Upper", "BB_Middle", "BB_Lower"])
        src["BandWidth"] = pd.to_numeric(src["BandWidth"], errors="coerce").round(4)
        src["BW_Threshold"] = pd.to_numeric(src["BW_Threshold"], errors="coerce").round(4)
        series.extend([
            {
                "name": "布林上轨",
                "type": "line",
                "showSymbol": False,
                "encode": {"x": "Date", "y": "BB_Upper"},
                "lineStyle": {"width": 1.5, "color": "#38BDF8"},
                "itemStyle": {"color": "#38BDF8"},
                "tooltip": {"valueFormatter": "dollar"},
            },
            {
                "name": "20日均线 (中轨)",
                "type": "line",
                "showSymbol": False,
                "encode": {"x": "Date", "y": "BB_Middle"},
                "lineStyle": {"width": 1.8, "color": "#F59E0B"},
                "itemStyle": {"color": "#F59E0B"},
                "tooltip": {"valueFormatter": "dollar"},
            },
            {
                "name": "布林下轨",
                "type": "line",
                "showSymbol": False,
                "encode": {"x": "Date", "y": "BB_Lower"},
                "lineStyle": {"width": 1.5, "color": "#38BDF8"},
                "itemStyle": {"color": "#38BDF8"},
                "tooltip": {"valueFormatter": "dollar"},
            },
        ])

    elif code == "S06" and "SMA_Fast" in src.columns:
        keep_cols.extend(["SMA_Fast", "SMA_Mid", "SMA_Slow"])
        round_targets.extend(["SMA_Fast", "SMA_Mid", "SMA_Slow"])
        series.extend([
            {
                "name": "SMA 20",
                "type": "line",
                "showSymbol": False,
                "encode": {"x": "Date", "y": "SMA_Fast"},
                "lineStyle": {"width": 1.8, "color": "#38BDF8"},
                "itemStyle": {"color": "#38BDF8"},
                "tooltip": {"valueFormatter": "dollar"},
            },
            {
                "name": "SMA 50",
                "type": "line",
                "showSymbol": False,
                "encode": {"x": "Date", "y": "SMA_Mid"},
                "lineStyle": {"width": 1.8, "color": "#F59E0B"},
                "itemStyle": {"color": "#F59E0B"},
                "tooltip": {"valueFormatter": "dollar"},
            },
            {
                "name": "SMA 200",
                "type": "line",
                "showSymbol": False,
                "encode": {"x": "Date", "y": "SMA_Slow"},
                "lineStyle": {"width": 1.8, "color": "#A855F7"},
                "itemStyle": {"color": "#A855F7"},
                "tooltip": {"valueFormatter": "dollar"},
            },
        ])

    # Buy/Sell triangle scatter markers (only visible if trades exist)
    if result.trades:
        series.extend([
            {
                "name": "买入信号 (Buy)",
                "type": "scatter",
                "symbol": "triangle",
                "symbolSize": 14,
                "encode": {"x": "Date", "y": "BuySignal"},
                "itemStyle": {"color": "#22C55E"},
                "tooltip": {"valueFormatter": "dollar"},
            },
            {
                "name": "卖出信号 (Sell)",
                "type": "scatter",
                "symbol": "triangle",
                "symbolRotate": 180,
                "symbolSize": 14,
                "encode": {"x": "Date", "y": "SellSignal"},
                "itemStyle": {"color": "#EF4444"},
                "tooltip": {"valueFormatter": "dollar"},
            },
        ])

    src = _round_cols(src, round_targets, 2)
    dataset_df = src[keep_cols].copy()

    if has_sub_panel:
        data_zoom, toolbox = _base_zoom_and_toolbox([0, 1])
        grids = [
            {"left": 18, "right": 18, "top": 64, "height": "48%", "containLabel": True},
            {"left": 18, "right": 18, "top": "64%", "height": "20%", "containLabel": True},
        ]
        x_axes = [
            {
                "type": "category",
                "boundaryGap": True,
                "axisLine": {"onZero": False},
                "axisLabel": {"show": False},
                "axisTick": {"show": False},
            },
            {
                "type": "category",
                "gridIndex": 1,
                "boundaryGap": True,
                "axisLine": {"onZero": False},
                "axisLabel": {"show": True},
            },
        ]
        if code == "S02":
            y_axes = [
                {"scale": True, "axisLabel": {"formatter": "${value}"}},
                {"gridIndex": 1, "min": 0, "max": 100, "splitNumber": 2},
            ]
            series.append({
                "name": "RSI(14)",
                "type": "line",
                "xAxisIndex": 1,
                "yAxisIndex": 1,
                "showSymbol": False,
                "encode": {"x": "Date", "y": "RSI"},
                "lineStyle": {"width": 1.8, "color": "#38BDF8"},
                "itemStyle": {"color": "#38BDF8"},
                "markLine": {
                    "symbol": "none",
                    "silent": True,
                    "data": [
                        {"yAxis": 30, "lineStyle": {"color": "#22C55E", "type": "dashed"}},
                        {"yAxis": 70, "lineStyle": {"color": "#EF4444", "type": "dashed"}},
                    ],
                },
            })
        else:  # S05 BandWidth Squeeze
            y_axes = [
                {"scale": True, "axisLabel": {"formatter": "${value}"}},
                {"scale": True, "gridIndex": 1, "splitNumber": 2},
            ]
            series.extend([
                {
                    "name": "布林带宽度 (BW)",
                    "type": "line",
                    "xAxisIndex": 1,
                    "yAxisIndex": 1,
                    "showSymbol": False,
                    "encode": {"x": "Date", "y": "BandWidth"},
                    "lineStyle": {"width": 1.8, "color": "#38BDF8"},
                    "itemStyle": {"color": "#38BDF8"},
                },
                {
                    "name": "挤压阈值 (最低10%)",
                    "type": "line",
                    "xAxisIndex": 1,
                    "yAxisIndex": 1,
                    "showSymbol": False,
                    "encode": {"x": "Date", "y": "BW_Threshold"},
                    "lineStyle": {"width": 1.5, "type": "dashed", "color": "#F59E0B"},
                    "itemStyle": {"color": "#F59E0B"},
                },
            ])
        height_px = 520
    else:
        data_zoom, toolbox = _base_zoom_and_toolbox([0])
        grids = [{"left": 18, "right": 18, "top": 64, "bottom": 66, "containLabel": True}]
        x_axes = [{"type": "category", "boundaryGap": True, "axisLine": {"onZero": False}}]
        y_axes = [{"scale": True, "axisLabel": {"formatter": "${value}"}}]
        height_px = 440

    option = {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
        "legend": {"top": 30, "left": "center"},
        "toolbox": toolbox,
        "axisPointer": {"link": [{"xAxisIndex": "all"}]},
        "grid": grids,
        "xAxis": x_axes,
        "yAxis": y_axes,
        "dataZoom": data_zoom,
        "dataset": {"source": dataset_df},
        "series": series,
    }
    subtitle = f"{result.ticker} K线与 {result.strategy_name} 信号诊断附图"
    return [(subtitle, option, height_px, subtitle)]


def _build_portfolio_diagnosis_chart(result) -> Tuple[Dict[str, Any], int, str]:
    """Builds normalized Strategy vs Benchmark chart with Rebalance trigger vertical lines & zoom slider."""
    eq_df = result.equity_df.copy().reset_index()
    date_col = eq_df.columns[0]
    eq_df = eq_df.rename(columns={date_col: "Date"})
    eq_df["Date"] = pd.to_datetime(eq_df["Date"]).dt.strftime("%Y-%m-%d")

    eq_df["StrategyNorm"] = (eq_df["Strategy_Equity"] / eq_df["Strategy_Equity"].iloc[0] * 100.0).round(2)
    eq_df["BenchmarkNorm"] = (eq_df["Benchmark_Equity"] / eq_df["Benchmark_Equity"].iloc[0] * 100.0).round(2)
    dataset_df = eq_df[["Date", "StrategyNorm", "BenchmarkNorm"]].copy()

    # Rebalance event vertical lines on exact category date strings
    mark_lines = []
    events = getattr(result, "rebalance_events", [])
    date_set = set(dataset_df["Date"].tolist())
    for ev in events:
        d_str = ev.date.strftime("%Y-%m-%d")
        if d_str in date_set:
            mark_lines.append({"xAxis": d_str, "lineStyle": {"color": "#F59E0B", "type": "dashed"}})

    data_zoom, toolbox = _base_zoom_and_toolbox([0])
    strat_series: Dict[str, Any] = {
        "name": f"组合策略 ({result.strategy_name})",
        "type": "line",
        "showSymbol": False,
        "encode": {"x": "Date", "y": "StrategyNorm"},
        "lineStyle": {"width": 2.4, "color": "#10B981"},
        "itemStyle": {"color": "#10B981"},
    }
    if mark_lines:
        strat_series["markLine"] = {"symbol": "none", "silent": True, "label": {"show": False}, "data": mark_lines}

    option = {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
        "legend": {"top": 30, "left": "center"},
        "toolbox": toolbox,
        "grid": [{"left": 18, "right": 18, "top": 64, "bottom": 66, "containLabel": True}],
        "xAxis": [{"type": "category", "boundaryGap": False}],
        "yAxis": [{"scale": True, "name": "归一化净值 (基期=100)"}],
        "dataZoom": data_zoom,
        "dataset": {"source": dataset_df},
        "series": [
            strat_series,
            {
                "name": f"对比基准 ({getattr(result, 'benchmark_name', 'Benchmark')})",
                "type": "line",
                "showSymbol": False,
                "encode": {"x": "Date", "y": "BenchmarkNorm"},
                "lineStyle": {"width": 2.0, "type": "dashed", "color": "#38BDF8"},
                "itemStyle": {"color": "#38BDF8"},
            },
        ],
    }
    return option, 400, f"{result.strategy_name} portfolio diagnosis chart"
