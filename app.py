#!/usr/bin/env python3
"""
Streamlit Web UI for U.S. Equity Trading Strategy Backtesting System (v2.2)
美股量化交易与资产配置回测系统 - 交互式可视化控制台
"""

import os
from typing import Dict, Any, List, Tuple
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="美股量化交易策略回测系统 v2.2",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from backtester.data_loader import DataLoader
from backtester.engine import BacktestEngine
from backtester.portfolio_engine import PortfolioEngine
from backtester.visualizer import Visualizer
from backtester.portfolio_visualizer import PortfolioVisualizer
from backtester.risk_analyzer import assess_investment_risk
from backtester.universe import NASDAQ_TOP_50_TICKERS, NASDAQ_TOP_50_MAP
from backtester.trade_diagnosis import diagnose_backtest_trades
from backtester.chart_ui import (
    build_period_candlestick_chart,
    build_diagnosis_candlestick_chart,
)

# Single stock strategies
from backtester.strategies.ma_crossover import DualMACrossoverStrategy
from backtester.strategies.rsi_strategy import RSIStrategy
from backtester.strategies.donchian_breakout import DonchianBreakoutStrategy
from backtester.strategies.bollinger_squeeze import BollingerBandSqueezeStrategy
from backtester.strategies.trend_pullback import TrendPullbackStrategy

# Portfolio / Multi-asset strategies
from backtester.strategies.relative_momentum import RelativeMomentumRotation
from backtester.strategies.threshold_rebalancing import ThresholdRebalancing
from backtester.strategies.quality_value_factor import QualityValueFactorStrategy, DEFAULT_FACTOR_UNIVERSE
from backtester.strategies.dual_momentum import DualMomentumStrategy


# Ordered strictly from S01 to S09
STRATEGY_DETAILS = {
    "S01": {
        "name": "S01 - 双均线交叉策略 (Dual MA Crossover)",
        "tag": "单股",
        "type": "single",
        "genre": "趋势跟踪",
        "desc": "短期均线 (20 SMA) 向上金叉中期均线 (50 SMA) 全仓买入；死叉卖出清仓避险。适合捕捉单边中长期牛市行情。",
    },
    "S02": {
        "name": "S02 - RSI 均值回归策略 (RSI Mean Reversion)",
        "tag": "单股",
        "type": "single",
        "genre": "均值回归",
        "desc": "以 200 日线判断大趋势；当 14 日 RSI < 30 超卖且位于 200 日线上逢低买入，RSI > 70 触顶获利了结。",
    },
    "S03": {
        "name": "S03 - 唐奇安通道突破策略 (Donchian Breakout)",
        "tag": "单股",
        "type": "single",
        "genre": "突破交易",
        "desc": "突破过去 20 日最高价买入进场；跌破过去 10 日最低价止损/卖出。海龟交易法则收盘确认版，用追高换取趋势确认。",
    },
    "S04": {
        "name": "S04 - 相对动量轮动策略 (Relative Momentum Rotation)",
        "tag": "组合",
        "type": "portfolio",
        "genre": "横截面动量",
        "desc": "每月最后一个交易日按 12-1 动量排序股票池，精选表现最强的前 4 只正动量股票各配 20%，保留至少 20% 防御现金。",
    },
    "S05": {
        "name": "S05 - 布林带挤压突破策略 (Bollinger Squeeze)",
        "tag": "单股",
        "type": "single",
        "genre": "波动率突破",
        "desc": "捕捉波动率周期切换：当带宽处于半年低位（窄幅盘整），且 5 日内收盘放量向上突破上轨时买入，跌破 20 日均线离场。",
    },
    "S06": {
        "name": "S06 - 趋势回撤买入策略 (Trend Pullback)",
        "tag": "单股",
        "type": "single",
        "genre": "趋势回撤",
        "desc": "顺大势逆小势：50 SMA > 200 SMA 确认大牛市；等待短期回调触及 50 SMA；今日收盘重新站上 20 SMA 时右侧上车。",
    },
    "S07": {
        "name": "S07 - 阈值动态再平衡策略 (Threshold Rebalancing)",
        "tag": "组合",
        "type": "portfolio",
        "genre": "组合风险控制",
        "desc": "设定资产目标权重与容忍带 (如 ±25%)；月度动态检测，大涨超标时止盈减仓，超跌低配时机械补仓，守住 15%~20% 现金底线。",
    },
    "S08": {
        "name": "S08 - 质量与价值因子策略 (Quality & Value Factor)",
        "tag": "组合",
        "type": "portfolio",
        "genre": "基本面多因子",
        "desc": "季频财报筛选：ROIC > 10%, 自由现金流 > 0, 净负债/EBITDA < 3；按自由现金流收益率 (FCF Yield) 排序买入 Top 5，各 15%，保留 25% 现金。",
    },
    "S09": {
        "name": "S09 - 双重动量资产轮动策略 (Dual Momentum)",
        "tag": "宏观组合",
        "type": "portfolio",
        "genre": "防御性资产轮动",
        "desc": "Gary Antonacci 经典模型：月度比较美股 (SPY) 与全球除美 (VEU) 12 个月收益，胜者若跑赢短期美债 (BIL) 则持有胜者，否则 100% 撤回国债避险。",
    },
}

TIMEFRAME_OPTIONS = {
    "最近 3 年 (3y - 推荐)": {"eval_days": 756, "years": 3},
    "最近 5 年 (5y - 穿越牛熊)": {"eval_days": 1260, "years": 5},
    "最近 1 年 (1y - 年度回顾)": {"eval_days": 252, "years": 1},
    "最近半年 (6m - 中期波段)": {"eval_days": 126, "years": 1},
    "最近 1 个月 (1m - 短线月度)": {"eval_days": 21, "years": 1},
    "最近 1 周 (1w - 5个交易日)": {"eval_days": 5, "years": 1},
    "最近 3 天 (3d - 极短线事件)": {"eval_days": 3, "years": 1},
}


@st.cache_data(ttl=1800, show_spinner=False)
def load_period_ohlcv(ticker: str, years: int, eval_days: int) -> pd.DataFrame:
    """Loads OHLCV market data for the chosen ticker and slices it to the selected period."""
    clean_t = ticker.strip().upper()
    if clean_t in {"SPACEX", "SPACE-X", "SPACE_X"}:
        clean_t = "SPCX"
    df = DataLoader().load_data(ticker=clean_t, lookback_years=years)
    if eval_days and 0 < eval_days < len(df):
        return df.iloc[-eval_days:].copy()
    return df


def execute_backtest(
    selected_code: str,
    target_ticker: str,
    selected_universe: List[str],
    years: int,
    eval_days: int,
    initial_capital: float,
    slippage_bps: float,
) -> Tuple[Any, Dict[str, Any], bytes, bool]:
    """
    Runs the selected strategy backtest and returns:
    (result, risk_profile, report_png_bytes, is_portfolio)
    Note: Renders high-res chart bytes directly in memory for instant web display without writing to disk!
    """
    loader = DataLoader()
    strat_meta = STRATEGY_DETAILS[selected_code]

    if strat_meta["type"] == "single":
        ticker = target_ticker.strip().upper()
        if ticker in {"SPACEX", "SPACE-X", "SPACE_X"}:
            ticker = "SPCX"

        df = loader.load_data(ticker=ticker, lookback_years=years)

        if selected_code == "S01":
            strategy = DualMACrossoverStrategy(short_window=20, long_window=50)
        elif selected_code == "S02":
            strategy = RSIStrategy(
                period=14,
                oversold=30.0,
                overbought=70.0,
                use_trend_filter=True,
                trend_ma_period=200,
            )
        elif selected_code == "S03":
            strategy = DonchianBreakoutStrategy(entry_window=20, exit_window=10)
        elif selected_code == "S05":
            strategy = BollingerBandSqueezeStrategy(
                period=20,
                std_dev=2.0,
                squeeze_lookback=126,
                squeeze_percentile=10.0,
                squeeze_recent_window=5,
            )
        elif selected_code == "S06":
            strategy = TrendPullbackStrategy(
                sma_fast=20,
                sma_mid=50,
                sma_slow=200,
                pullback_window=5,
            )
        else:
            strategy = DualMACrossoverStrategy(short_window=20, long_window=50)

        engine = BacktestEngine(initial_capital=initial_capital, slippage_bps=slippage_bps)
        result = engine.run(df=df, strategy=strategy, ticker=ticker, eval_days=eval_days)
        risk_profile = assess_investment_risk(result.metrics, ticker=ticker)
        png_bytes = Visualizer.render_png_bytes(result)
        return result, risk_profile, png_bytes, False

    else:
        p_engine = PortfolioEngine(initial_capital=initial_capital, slippage_bps=slippage_bps)

        if selected_code == "S09":
            prices_df = loader.load_multi_data(["SPY", "VEU", "BIL"], lookback_years=years)
            dm = DualMomentumStrategy()
            weights_df = dm.compute_target_weights(prices_df)
            result = p_engine.run(
                prices_df=prices_df,
                target_weights_df=weights_df,
                strategy_name="S09_DualMomentum",
                benchmark_series=prices_df["SPY"],
                benchmark_name="SPY (US Equities)",
                eval_days=eval_days,
            )
        elif selected_code == "S04":
            prices_df = loader.load_multi_data(selected_universe, lookback_years=years)
            rm = RelativeMomentumRotation(top_n=4, target_weight_per_stock=0.20)
            weights_df = rm.compute_target_weights(prices_df)
            norm_prices = prices_df.divide(prices_df.iloc[0])
            benchmark_series = norm_prices.mean(axis=1) * prices_df.iloc[0].mean()
            result = p_engine.run(
                prices_df=prices_df,
                target_weights_df=weights_df,
                strategy_name="S04_RelativeMomentumRotation",
                benchmark_series=benchmark_series,
                benchmark_name="Equal-Weight Buy & Hold",
                eval_days=eval_days,
            )
        elif selected_code == "S07":
            prices_df = loader.load_multi_data(selected_universe, lookback_years=years)
            tw_per_stock = 0.80 / len(selected_universe)
            target_weights = {asset: tw_per_stock for asset in selected_universe}
            tr = ThresholdRebalancing(target_weights=target_weights, target_cash_pct=0.20, band_pct=0.25)
            result = tr.simulate(
                prices_df=prices_df,
                initial_capital=initial_capital,
                slippage_bps=slippage_bps,
                benchmark_name="Buy & Hold (No Rebalance)",
                eval_days=eval_days,
            )
        elif selected_code == "S08":
            prices_df = loader.load_multi_data(selected_universe, lookback_years=years)
            qf = QualityValueFactorStrategy(universe=selected_universe, top_n=5, target_weight_per_stock=0.15)
            weights_df = qf.compute_target_weights(prices_df)
            norm_prices = prices_df.divide(prices_df.iloc[0])
            benchmark_series = norm_prices.mean(axis=1) * prices_df.iloc[0].mean()
            result = p_engine.run(
                prices_df=prices_df,
                target_weights_df=weights_df,
                strategy_name="S08_QualityValueFactor",
                benchmark_series=benchmark_series,
                benchmark_name="Universe Equal-Weight",
                eval_days=eval_days,
            )
        else:
            raise ValueError(f"Unsupported portfolio strategy: {selected_code}")

        risk_profile = assess_investment_risk(result.metrics, ticker=result.strategy_name)
        png_bytes = PortfolioVisualizer.render_png_bytes(result)
        return result, risk_profile, png_bytes, True


# Initialize session state
if "last_run" not in st.session_state:
    st.session_state["last_run"] = None


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.title("⚙️ 量化控制台")
    st.markdown("---")

    # 1. Strategy selection (strict S01 -> S09 order)
    st.subheader("1. 选择交易策略")
    strategy_keys = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09"]
    strategy_labels = [
        f"{k} - {STRATEGY_DETAILS[k]['name'].split(' - ')[1].split(' (')[0]} [{STRATEGY_DETAILS[k]['tag']}]"
        for k in strategy_keys
    ]
    selected_label = st.selectbox("策略列表", strategy_labels, index=0, key="strategy_select")
    selected_code = selected_label.split(" - ")[0]
    strat_meta = STRATEGY_DETAILS[selected_code]

    st.info(strat_meta["desc"], icon="💡")

    st.markdown("---")

    # 2. Asset / Universe selection
    st.subheader("2. 选择研究标的 / 资产池")
    target_ticker = "NVDA"
    selected_universe = ["NVDA", "TSLA", "SPCX", "AAPL", "MSFT", "AMZN", "GOOGL", "META"]

    if strat_meta["type"] == "single":
        ticker_mode = st.radio("标的选择模式", ["从纳斯达克 50 强中选择", "自定义输入代码"], horizontal=True, key="ticker_mode")
        if ticker_mode == "从纳斯达克 50 强中选择":
            top50_options = [
                f"{sym} - {NASDAQ_TOP_50_MAP.get(sym, {}).get('name', sym)} ({NASDAQ_TOP_50_MAP.get(sym, {}).get('sector', '')})"
                for sym in NASDAQ_TOP_50_TICKERS
            ]
            chosen_top50 = st.selectbox("纳斯达克 Top 50 龙头", top50_options, index=0, key="top50_select")
            target_ticker = chosen_top50.split(" - ")[0]
        else:
            target_ticker = (
                st.text_input("请输入美股代码 (如 NVDA, TSLA, SPCX, AAPL)", value="NVDA", key="custom_ticker")
                .strip()
                .upper()
                or "NVDA"
            )
    else:
        if selected_code == "S09":
            st.markdown("🔒 **宏观核心大类资产固定为**：\n- **SPY** (美股标普500)\n- **VEU** (全球除美股票)\n- **BIL** (1-3月短期国债/现金)")
            selected_universe = ["SPY", "VEU", "BIL"]
        else:
            pool_choice = st.radio(
                "资产池范围",
                [
                    "核心精选科技池 (Mag 7 + SPCX)",
                    f"纳斯达克市值前 50 强池 (共 {len(NASDAQ_TOP_50_TICKERS)} 家龙头)",
                ],
                key="pool_choice",
            )
            if "50" in pool_choice:
                selected_universe = NASDAQ_TOP_50_TICKERS
            elif selected_code == "S08":
                selected_universe = DEFAULT_FACTOR_UNIVERSE
            else:
                selected_universe = ["NVDA", "TSLA", "SPCX", "AAPL", "MSFT", "AMZN", "GOOGL", "META"]

    st.markdown("---")

    # 3. Timeframe selection
    st.subheader("3. 回测时间周期")
    tf_name = st.radio("时间跨度", list(TIMEFRAME_OPTIONS.keys()), index=0, key="timeframe_radio")
    tf_info = TIMEFRAME_OPTIONS[tf_name]
    eval_days = tf_info["eval_days"]
    years = tf_info["years"]

    st.markdown("---")

    # 4. Capital & Slippage
    st.subheader("4. 资金与交易摩擦")
    default_cap = 10000.0 if strat_meta["type"] == "single" else 100000.0
    initial_capital = st.number_input(
        "初始资金 ($ USD)", min_value=1000.0, max_value=10000000.0, value=default_cap, step=5000.0, key="cap_input"
    )
    slippage_bps = st.slider("交易滑点 (BPS)", min_value=0, max_value=30, value=5, help="1 BPS = 0.01%", key="slippage_slider")

    st.markdown("---")
    run_btn = st.button("🚀 运行量化回测", type="primary", width="stretch", key="run_btn")


# ==========================================
# MAIN DASHBOARD AREA
# ==========================================
st.title("📈 美股量化交易策略回测与风控系统")
st.caption("基于严格真实行情 · 无前视偏差 · 充分预热切片 · 投资风险评级模型 (R1~R4) · v2.2")

# --- TOP HERO SECTION: PERIOD CANDLESTICK K-LINE CHART RIGHT BELOW TITLE ---
with st.container(border=True):
    if strat_meta["type"] == "single":
        hero_ticker = target_ticker
    else:
        col_h1, col_h2 = st.columns([2, 1])
        with col_h1:
            st.markdown("#### 🕯️ 标的区间 K 线行情预览 (Period Candlestick Chart)")
        with col_h2:
            hero_ticker = st.selectbox(
                "选择K线预览成分股",
                selected_universe,
                index=0,
                label_visibility="collapsed",
                key="hero_portfolio_ticker",
            )
    if strat_meta["type"] == "single":
        st.markdown(f"#### 🕯️ **{hero_ticker}** 区间 K 线行情预览 (Period: {tf_name})")

    try:
        hero_ohlcv = load_period_ohlcv(hero_ticker, years=years, eval_days=eval_days)
        hero_k_chart = build_period_candlestick_chart(hero_ohlcv, ticker=hero_ticker, period_label=tf_name)
        st.altair_chart(hero_k_chart, width="stretch", alt=f"{hero_ticker} period candlestick chart")
    except Exception as e:
        st.warning(f"⚠️ 无法加载 {hero_ticker} 的 K 线预览行情：{e}")


# Execute Backtest when button clicked
if run_btn:
    with st.spinner("⏳ 正在载入真实行情数据并执行量化仿真，请稍候..."):
        try:
            result, risk_profile, png_bytes, is_portfolio = execute_backtest(
                selected_code=selected_code,
                target_ticker=target_ticker,
                selected_universe=selected_universe,
                years=years,
                eval_days=eval_days,
                initial_capital=initial_capital,
                slippage_bps=slippage_bps,
            )
            diagnosis = diagnose_backtest_trades(result, selected_code, is_portfolio=is_portfolio)
            st.session_state["last_run"] = {
                "selected_code": selected_code,
                "result": result,
                "risk_profile": risk_profile,
                "png_bytes": png_bytes,
                "saved_local_path": None,
                "is_portfolio": is_portfolio,
                "diagnosis": diagnosis,
            }
        except Exception as e:
            st.error(f"❌ 回测运行发生错误: {str(e)}")

last_run = st.session_state["last_run"]

if not last_run:
    st.markdown("""
    ### 👋 欢迎使用量化回测控制台！
    上方为您已选定标的与时间周期的**实时日 K 线与成交量视图**。

    #### 💡 快速上手步骤：
    1. 在**左侧控制台**挑选您想要检验的交易策略（按 S01~S09 排序）；
    2. 选择目标股票（内置**纳斯达克市值前 50 强标的**，含 SpaceX `SPCX`、特斯拉 `TSLA`、英伟达 `NVDA` 等）；
    3. 选择回测时间档位（支持 **3天、1周、1个月、半年、1年、3年、5年**）；
    4. 点击左下角 **【🚀 运行量化回测】**，即可查看**风险评级卡**、**交易触发原因诊断**、**K 线信号图解**与**收益报表**！
    """)

    with st.expander("🔍 展开查看纳斯达克市值前 50 强标的速查表 (Top 50 Companies)"):
        top50_df = pd.DataFrame([
            {"代码": sym, "公司名称": data["name"], "所属板块": data["sector"]}
            for sym, data in NASDAQ_TOP_50_MAP.items()
        ])
        st.dataframe(top50_df, width="stretch", height=350)
else:
    result = last_run["result"]
    risk_profile = last_run["risk_profile"]
    is_portfolio = last_run["is_portfolio"]
    run_code = last_run["selected_code"]
    diagnosis = last_run["diagnosis"]
    png_bytes = last_run["png_bytes"]

    # --- SECTION 1: INVESTMENT RISK ASSESSMENT CARD ---
    st.markdown("### 🛡️ 投资风险评级与初学者风控指南 (Risk Assessment Card)")

    level = risk_profile["risk_level"]
    score = risk_profile["risk_score"]
    mdd = risk_profile["max_drawdown"]
    vol = risk_profile["volatility"]
    sharpe = risk_profile["sharpe_ratio"]

    badge_colors = {
        "LOW_RISK": ("#064E3B", "#34D399"),
        "MODERATE_RISK": ("#713F12", "#FBBF24"),
        "HIGH_RISK": ("#7C2D12", "#FB923C"),
        "EXTREME_RISK": ("#881337", "#F43F5E"),
    }
    bg_c, fg_c = badge_colors.get(risk_profile["badge"], ("#713F12", "#FBBF24"))

    col_risk1, col_risk2 = st.columns([1.2, 2])

    with col_risk1:
        with st.container(border=True):
            st.caption(f"综合投资风险等级 ({risk_profile['ticker']})")
            st.markdown(
                f"<span style='background-color:{bg_c}; color:{fg_c}; padding:4px 12px; border-radius:6px; font-weight:bold; font-size:18px;'>"
                f"{risk_profile['icon']} {level}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"综合风险评分：**{score:.1f} / 100**")
            st.progress(min(100, max(0, int(score))))
            st.markdown(
                f"- 最大回撤 (MDD)：**{mdd*100:.2f}%**\n"
                f"- 年化波动率 (Vol)：**{vol*100:.2f}%**\n"
                f"- 夏普比率 (Sharpe)：**{sharpe:.2f}**"
            )

    with col_risk2:
        with st.container(border=True):
            st.markdown("#### 🎯 初学者操作风控红线 (Disciplined Guidelines)")
            st.markdown(
                f"- 📌 **建议单股仓位上限**：资金占比控制在 **【{risk_profile['max_position']}】** 以内\n"
                f"- 🛑 **防守止损红线**：跌破进场价 **【{risk_profile['stop_loss']}】** 必须坚决执行防守减仓或离场\n"
                f"- 🧭 **资产适配定位**：{risk_profile['style']}\n"
                f"- 🧘 **交易心理建设**：{risk_profile['psychology']}"
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- SECTION 2: METRICS CARDS ---
    st.markdown("### 📊 核心量化指标看板 (Key Performance Metrics)")
    m = result.metrics

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="最终账户资产 (Final Equity)",
            value=f"${result.final_equity:,.2f}",
            delta=f"{m['Strategy Total Return']*100:+.2f}% 累计收益",
        )
    with col2:
        strat_cagr = m["Strategy CAGR"] * 100
        bench_cagr = m["Benchmark CAGR"] * 100
        st.metric(
            label="年化收益率 (CAGR)",
            value=f"{strat_cagr:+.2f}%",
            delta=f"{strat_cagr - bench_cagr:+.2f}% 超额",
        )
    with col3:
        strat_sharpe = m["Strategy Sharpe"]
        st.metric(
            label="夏普比率 (Sharpe Ratio)",
            value=f"{strat_sharpe:.2f}",
            delta=f"基准: {m['Benchmark Sharpe']:.2f}",
        )
    with col4:
        strat_mdd = m["Strategy Max Drawdown"] * 100
        bench_mdd = m["Benchmark Max Drawdown"] * 100
        st.metric(
            label="历史最大回撤 (MDD)",
            value=f"{strat_mdd:.2f}%",
            delta=f"基准: {bench_mdd:.2f}%",
            delta_color="inverse",
        )

    col_sub1, col_sub2, col_sub3, col_sub4 = st.columns(4)
    with col_sub1:
        st.metric("年化波动率 (Volatility)", f"{m['Strategy Volatility']*100:.2f}%")
    with col_sub2:
        if not is_portfolio:
            st.metric("已完成交易次数", f"{m.get('Total Trades', 0)} 笔")
        else:
            st.metric("资产再平衡次数", f"{len(getattr(result, 'rebalance_events', []))} 次")
    with col_sub3:
        if not is_portfolio:
            st.metric("交易胜率 (Win Rate)", f"{m.get('Win Rate', 0)*100:.2f}%")
        else:
            st.metric("基准资产", getattr(result, "benchmark_name", "Buy & Hold"))
    with col_sub4:
        if not is_portfolio:
            pf = m.get("Profit Factor", 0)
            st.metric("盈亏比 (Profit Factor)", f"{pf:.2f}" if pf != float("inf") else "∞")
        else:
            st.metric("评估窗口", f"{result.start_date} ~ {result.end_date}")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- SECTION 3: DEDICATED TRADE EXECUTION DIAGNOSIS ---
    st.markdown("### 🔎 交易次数诊断与逻辑拆解 (Trade Execution Diagnosis)")
    with st.container(border=True):
        st.markdown(f"#### {diagnosis['headline']}")
        for bullet in diagnosis["bullets"]:
            st.markdown(f"- {bullet}")

        if diagnosis["condition_rows"]:
            st.markdown("**📌 策略条件触发情况统计明细：**")
            st.table(pd.DataFrame(diagnosis["condition_rows"]), width="stretch")

        if not diagnosis["events_df"].empty:
            with st.expander(f"📑 点击展开详细交易 / 调仓触发流水记录（共 {len(diagnosis['events_df'])} 条）"):
                st.dataframe(diagnosis["events_df"], width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- SECTION 4: DIAGNOSTIC K-LINE & INDICATOR TRIGGER CHART ---
    st.markdown("### 🕯️ 策略触发条件与 K 线图解 (Signal & Condition Visual Diagnosis)")
    with st.container(border=True):
        st.caption("通过 K 线与策略指标线（如均线、RSI 超卖线与200日牛熊过滤线、唐奇安通道、布林带挤压线等）直观展示交易点为何触发或为何被拦截。")
        diag_k_chart = build_diagnosis_candlestick_chart(result, run_code, is_portfolio=is_portfolio)
        st.altair_chart(diag_k_chart, width="stretch", alt=f"{run_code} diagnostic candlestick chart")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- SECTION 5: WEB REPORT VISUALIZATION FIRST -> THEN ASK DOWNLOAD / SAVE ---
    st.markdown("### 📈 全景回测分析报告图表 (Performance Report)")
    st.image(
        png_bytes,
        caption=f"策略回测全景报表: {result.strategy_name} ({result.start_date} 至 {result.end_date})",
        alt=f"{result.strategy_name} performance report chart",
        width="stretch",
    )

    with st.container(border=True):
        st.info(
            "🙋 **图表已在网页为您即时生成展示。** 请确认是否需要将该高清分析图表下载到您的电脑本地，或保存到项目的 `reports/` 文件夹？",
            icon="💾",
        )
        col_dl1, col_dl2, col_dl3 = st.columns([1.4, 1.4, 2])
        clean_name = result.strategy_name.replace(" ", "_")
        target_label = getattr(result, "ticker", clean_name)
        default_png_name = f"{target_label}_{clean_name}_{result.start_date}_{result.end_date}.png"

        with col_dl1:
            st.download_button(
                label="📥 下载高清图表到本地电脑 (.png)",
                data=png_bytes,
                file_name=default_png_name,
                mime="image/png",
                width="stretch",
                key="download_png_btn",
            )
        with col_dl2:
            if st.button("📂 保存一份到项目 reports/ 目录", width="stretch", key="save_reports_btn"):
                reports_dir = os.path.join(os.path.dirname(__file__), "reports")
                os.makedirs(reports_dir, exist_ok=True)
                save_file = os.path.join(reports_dir, default_png_name)
                with open(save_file, "wb") as f:
                    f.write(png_bytes)
                st.session_state["last_run"]["saved_local_path"] = save_file
                st.rerun()

        if last_run.get("saved_local_path"):
            st.success(f"✅ 已保存至项目目录：`{last_run['saved_local_path']}`")

    # --- SECTION 6: DATA TABLE & CSV EXPORT ---
    with st.expander("📋 查看完整回测对比明细与导出 CSV 数据"):
        strat_hdr = f"策略 ({result.strategy_name})"
        bench_hdr = f"基准 ({getattr(result, 'benchmark_name', 'Buy & Hold')})"

        table_rows = [
            {"指标 / Metric": "初始本金 (Initial Capital)", strat_hdr: f"${result.initial_capital:,.2f}", bench_hdr: f"${result.initial_capital:,.2f}"},
            {"指标 / Metric": "期末资产 (Final Equity)", strat_hdr: f"${result.final_equity:,.2f}", bench_hdr: f"${result.benchmark_final_equity:,.2f}"},
            {"指标 / Metric": "累计总收益率 (Cumulative Return)", strat_hdr: f"{m['Strategy Total Return']*100:+.2f}%", bench_hdr: f"{m['Benchmark Total Return']*100:+.2f}%"},
            {"指标 / Metric": "年化复合增长率 (CAGR)", strat_hdr: f"{m['Strategy CAGR']*100:+.2f}%", bench_hdr: f"{m['Benchmark CAGR']*100:+.2f}%"},
            {"指标 / Metric": "年化波动率 (Volatility)", strat_hdr: f"{m['Strategy Volatility']*100:.2f}%", bench_hdr: f"{m['Benchmark Volatility']*100:.2f}%"},
            {"指标 / Metric": "夏普比率 (Sharpe Ratio Rf=4%)", strat_hdr: f"{m['Strategy Sharpe']:.2f}", bench_hdr: f"{m['Benchmark Sharpe']:.2f}"},
            {"指标 / Metric": "历史最大回撤 (Max Drawdown)", strat_hdr: f"{m['Strategy Max Drawdown']*100:.2f}%", bench_hdr: f"{m['Benchmark Max Drawdown']*100:.2f}%"},
        ]

        st.table(pd.DataFrame(table_rows), width="stretch")

        csv_data = result.equity_df.to_csv().encode("utf-8")
        st.download_button(
            label="📥 下载回测每日净值走势明细 (CSV)",
            data=csv_data,
            file_name=f"backtest_{run_code}_{result.start_date}_{result.end_date}.csv",
            mime="text/csv",
            key="download_csv_btn",
        )
