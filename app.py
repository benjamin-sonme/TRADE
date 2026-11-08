#!/usr/bin/env python3
"""
Streamlit Web UI for U.S. Equity Trading Strategy Backtesting System (v2.1)
美股量化交易与资产配置回测系统 - 交互式可视化控制台
"""

import os
import sys
import pandas as pd
import streamlit as st

# Set page config
st.set_page_config(
    page_title="美股量化交易策略回测系统 v2.1",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E293B;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #334155;
        margin-bottom: 15px;
    }
    .risk-badge-r1 { background-color: #064E3B; color: #34D399; padding: 4px 12px; border-radius: 6px; font-weight: bold; }
    .risk-badge-r2 { background-color: #713F12; color: #FBBF24; padding: 4px 12px; border-radius: 6px; font-weight: bold; }
    .risk-badge-r3 { background-color: #7C2D12; color: #FB923C; padding: 4px 12px; border-radius: 6px; font-weight: bold; }
    .risk-badge-r4 { background-color: #881337; color: #F43F5E; padding: 4px 12px; border-radius: 6px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

from backtester.data_loader import DataLoader
from backtester.engine import BacktestEngine
from backtester.portfolio_engine import PortfolioEngine
from backtester.visualizer import Visualizer
from backtester.portfolio_visualizer import PortfolioVisualizer
from backtester.risk_analyzer import assess_investment_risk
from backtester.universe import NASDAQ_TOP_50_TICKERS, NASDAQ_TOP_50_MAP

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


STRATEGY_DETAILS = {
    "S01": {
        "name": "S01. 双均线交叉策略 (Dual MA Crossover)",
        "type": "single",
        "genre": "趋势跟踪",
        "desc": "短期均线 (20 SMA) 向上金叉中期均线 (50 SMA) 全仓买入；死叉卖出清仓避险。适合捕捉单边中长期牛市行情。",
    },
    "S02": {
        "name": "S02. RSI 均值回归策略 (RSI Mean Reversion)",
        "type": "single",
        "genre": "均值回归",
        "desc": "以 200 日线判断大趋势；当 14 日 RSI < 30 超卖且位于 200 日线上逢低买入，RSI > 70 触顶获利了结。",
    },
    "S03": {
        "name": "S03. 唐奇安通道突破策略 (Donchian Breakout)",
        "type": "single",
        "genre": "突破交易",
        "desc": "突破过去 20 日最高价买入进场；跌破过去 10 日最低价止损/卖出。海龟交易法则收盘确认版，用追高换取趋势确认。",
    },
    "S05": {
        "name": "S05. 布林带挤压突破策略 (Bollinger Squeeze)",
        "type": "single",
        "genre": "波动率突破",
        "desc": "捕捉波动率周期切换：当带宽处于半年低位（窄幅盘整），且 5 日内收盘放量向上突破上轨时买入，跌破 20 日均线离场。",
    },
    "S06": {
        "name": "S06. 趋势回撤买入策略 (Trend Pullback)",
        "type": "single",
        "genre": "趋势回撤",
        "desc": "顺大势逆小势：50 SMA > 200 SMA 确认大牛市；等待短期回调触及 50 SMA；今日收盘重新站上 20 SMA 时右侧上车。",
    },
    "S04": {
        "name": "S04. 相对动量轮动策略 (Relative Momentum Rotation)",
        "type": "portfolio",
        "genre": "横截面动量",
        "desc": "每月最后一个交易日按 12-1 动量排序股票池，精选表现最强的前 4 只正动量股票各配 20%，保留至少 20% 防御现金。",
    },
    "S07": {
        "name": "S07. 阈值动态再平衡策略 (Threshold Rebalancing)",
        "type": "portfolio",
        "genre": "组合风险控制",
        "desc": "设定资产目标权重与容忍带 (如 ±25%)；月度动态检测，大涨超标时止盈减仓，超跌低配时机械补仓，守住 15%~20% 现金底线。",
    },
    "S08": {
        "name": "S08. 质量与价值因子策略 (Quality & Value Factor)",
        "type": "portfolio",
        "genre": "基本面多因子",
        "desc": "季频财报筛选：ROIC > 10%, 自由现金流 > 0, 净负债/EBITDA < 3；按自由现金流收益率 (FCF Yield) 排序买入 Top 5，各 15%，保留 25% 现金。",
    },
    "S09": {
        "name": "S09. 双重动量资产轮动策略 (Dual Momentum)",
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


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.title("⚙️ 量化控制台")
    st.markdown("---")

    # 1. Strategy selection
    st.subheader("1. 选择交易策略")
    strategy_keys = list(STRATEGY_DETAILS.keys())
    strategy_labels = [f"{k} - {STRATEGY_DETAILS[k]['name'].split(' ')[1]}" for k in strategy_keys]
    selected_label = st.selectbox("策略列表", strategy_labels, index=0)
    selected_code = selected_label.split(" - ")[0]
    strat_meta = STRATEGY_DETAILS[selected_code]

    # Strategy info box
    st.info(f"**【{strat_meta['genre']}】**\n\n{strat_meta['desc']}")

    st.markdown("---")

    # 2. Asset / Universe selection
    st.subheader("2. 选择研究标的 / 资产池")
    if strat_meta["type"] == "single":
        ticker_mode = st.radio("标的选择模式", ["从纳斯达克 50 强中选择", "自定义输入代码"], horizontal=True)
        if ticker_mode == "从纳斯达克 50 强中选择":
            top50_options = [
                f"{sym} - {NASDAQ_TOP_50_MAP.get(sym, {}).get('name', sym)} ({NASDAQ_TOP_50_MAP.get(sym, {}).get('sector', '')})"
                for sym in NASDAQ_TOP_50_TICKERS
            ]
            chosen_top50 = st.selectbox("纳斯达克 Top 50 龙头", top50_options, index=0)
            target_ticker = chosen_top50.split(" - ")[0]
        else:
            target_ticker = st.text_input("请输入美股代码 (如 NVDA, TSLA, SPCX, AAPL)", value="NVDA").strip().upper() or "NVDA"
    else:
        if selected_code == "S09":
            st.markdown("🔒 **宏观核心大类资产固定为**：\n- **SPY** (美股标普500)\n- **VEU** (全球除美股票)\n- **BIL** (1-3月短期国债/现金)")
            selected_universe = ["SPY", "VEU", "BIL"]
        else:
            pool_choice = st.radio(
                "资产池范围",
                [
                    "核心精选科技池 (Mag 7: NVDA, TSLA, SPCX, AAPL, MSFT, AMZN, GOOGL, META)",
                    f"纳斯达克市值前 50 强池 (共 {len(NASDAQ_TOP_50_TICKERS)} 家龙头企业)",
                ],
            )
            if "50" in pool_choice:
                selected_universe = NASDAQ_TOP_50_TICKERS
            else:
                selected_universe = ["NVDA", "TSLA", "SPCX", "AAPL", "MSFT", "AMZN", "GOOGL", "META"]

    st.markdown("---")

    # 3. Timeframe selection
    st.subheader("3. 回测时间周期")
    tf_name = st.radio("时间跨度", list(TIMEFRAME_OPTIONS.keys()), index=0)
    tf_info = TIMEFRAME_OPTIONS[tf_name]
    eval_days = tf_info["eval_days"]
    years = tf_info["years"]

    st.markdown("---")

    # 4. Capital & Slippage
    st.subheader("4. 资金与交易摩擦")
    default_cap = 10000.0 if strat_meta["type"] == "single" else 100000.0
    initial_capital = st.number_input("初始资金 ($ USD)", min_value=1000.0, max_value=10000000.0, value=default_cap, step=5000.0)
    slippage_bps = st.slider("交易滑点 (BPS)", min_value=0, max_value=30, value=5, help="1 BPS = 0.01%")

    st.markdown("---")
    run_btn = st.button("🚀 运行量化回测", type="primary", use_container_width=True)


# ==========================================
# MAIN DASHBOARD AREA
# ==========================================
st.title("📈 美股量化交易策略回测与风控系统")
st.caption("基于严格真实行情 · 无前视偏差 · 充分预热切片 · 投资风险评级模型 (R1~R4) · v2.1")

if not run_btn:
    # Landing Guide
    st.markdown("""
    ### 👋 欢迎使用量化回测控制台！
    本系统集成了经典趋势跟踪、均值回归、动量轮动与基本面多因子等 **9 大流派量化策略**。

    #### 💡 快速上手步骤：
    1. 在**左侧控制台**挑选您想要检验的交易策略；
    2. 选择目标股票（内置**纳斯达克市值前 50 强标的**，包含已上市的 SpaceX `SPCX`、特斯拉 `TSLA`、英伟达 `NVDA` 等）；
    3. 选择回测时间档位（涵盖 **3天、1周、1个月、半年、1年、3年、5年**）；
    4. 点击左下角的 **【🚀 运行量化回测】**，即刻生成多维度绩效对比、**企业投资风险评级卡 (R1~R4)** 及高清可视化图表！
    """)

    # Quick preview of Nasdaq Top 50
    with st.expander("🔍 展开查看纳斯达克市值前 50 强标的速查表 (Top 50 Companies)"):
        top50_df = pd.DataFrame([
            {"代码": sym, "公司名称": data["name"], "所属板块": data["sector"]}
            for sym, data in NASDAQ_TOP_50_MAP.items()
        ])
        st.dataframe(top50_df, use_container_width=True, height=350)

else:
    # Execute Backtest
    loader = DataLoader()

    with st.spinner("⏳ 正在载入真实行情数据并执行量化仿真，请稍候..."):
        try:
            if strat_meta["type"] == "single":
                # --- SINGLE STOCK PIPELINE ---
                df = loader.load_stock_data(ticker=target_ticker, lookback_years=years)

                # Instantiate strategy
                if selected_code == "S01":
                    strategy = DualMACrossoverStrategy(short_window=20, long_window=50)
                elif selected_code == "S02":
                    strategy = RSIStrategy(period=14, oversold=30, overbought=70, use_trend_filter=True, trend_window=200)
                elif selected_code == "S03":
                    strategy = DonchianBreakoutStrategy(entry_window=20, exit_window=10)
                elif selected_code == "S05":
                    strategy = BollingerBandSqueezeStrategy(period=20, num_std=2.0, lookback_bw=126, squeeze_quantile=0.10)
                elif selected_code == "S06":
                    strategy = TrendPullbackStrategy(short_window=20, mid_window=50, long_window=200, pullback_bars=5)
                else:
                    strategy = DualMACrossoverStrategy(short_window=20, long_window=50)

                engine = BacktestEngine(initial_capital=initial_capital, slippage_bps=slippage_bps)
                result = engine.run(df=df, strategy=strategy, ticker=target_ticker, eval_days=eval_days)
                risk_profile = assess_investment_risk(result.metrics, ticker=target_ticker)
                chart_path = Visualizer.plot_result(result)
                is_portfolio = False

            else:
                # --- PORTFOLIO PIPELINE ---
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
                        benchmark_name="Equal-Weight Buy & Hold",
                        eval_days=eval_days,
                    )

                risk_profile = assess_investment_risk(result.metrics, ticker=result.strategy_name)
                chart_path = PortfolioVisualizer.plot_result(result)
                is_portfolio = True

        except Exception as e:
            st.error(f"❌ 回测运行发生错误: {str(e)}")
            st.stop()

    # --- SECTION 1: INVESTMENT RISK ASSESSMENT CARD ---
    st.markdown("### 🛡️ 投资风险评级与初学者风控指南 (Risk Assessment Card)")

    level = risk_profile["level"]
    score = risk_profile["score"]
    factors = risk_profile["factors"]
    guidelines = risk_profile["guidelines"]

    badge_class = {
        "LOW_RISK": "risk-badge-r1",
        "MODERATE_RISK": "risk-badge-r2",
        "HIGH_RISK": "risk-badge-r3",
        "EXTREME_RISK": "risk-badge-r4",
    }.get(risk_profile["badge"], "risk-badge-r2")

    col_risk1, col_risk2 = st.columns([1.2, 2])

    with col_risk1:
        st.markdown(f"""
        <div style="background-color: #1E293B; border: 1px solid #334155; border-radius: 12px; padding: 20px;">
            <div style="font-size: 14px; color: #94A3B8;">综合投资风险等级</div>
            <div style="font-size: 24px; font-weight: bold; margin: 10px 0;">
                <span class="{badge_class}">{level}</span>
            </div>
            <div style="font-size: 13px; color: #CBD5E1; margin-top: 12px;">
                综合风险评分: <b>{score:.1f} / 100</b>
            </div>
            <div style="margin-top: 8px;">
                <div style="background-color: #334155; border-radius: 6px; height: 10px; width: 100%;">
                    <div style="background-color: {'#34D399' if score<=35 else ('#FBBF24' if score<=55 else ('#FB923C' if score<=75 else '#F43F5E'))}; height: 100%; border-radius: 6px; width: {min(100.0, score)}%;"></div>
                </div>
            </div>
            <hr style="border-color: #334155; margin: 15px 0;">
            <div style="font-size: 13px; color: #94A3B8;">
                • 最大回撤 (MDD): <b style="color: #F8FAFC;">{factors['Max Drawdown']*100:.2f}%</b><br>
                • 年化波动率: <b style="color: #F8FAFC;">{factors['Annualized Volatility']*100:.2f}%</b><br>
                • 夏普比率 (Sharpe): <b style="color: #F8FAFC;">{factors['Sharpe Ratio']:.2f}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_risk2:
        st.markdown(f"""
        <div style="background-color: #1E293B; border: 1px solid #334155; border-radius: 12px; padding: 20px; height: 100%;">
            <div style="font-size: 16px; font-weight: bold; color: #38BDF8; margin-bottom: 12px;">🎯 初学者操作风控红线 (Disciplined Guidelines)</div>
            <div style="font-size: 14px; color: #E2E8F0; line-height: 1.8;">
                📌 <b>建议仓位上限</b>：单标的资金占比建议控制在 <span style="color: #FBBF24; font-weight: bold;">【{guidelines['max_position_limit']}】</span> 以内<br>
                🛑 <b>防守止损红线</b>：跌破进场价 <span style="color: #F43F5E; font-weight: bold;">【{guidelines['defensive_stop_loss']}】</span> 必须坚决执行防守减仓或离场<br>
                🧭 <b>资产角色定位</b>：{guidelines['investment_style']}<br>
                🧘 <b>交易心理建设</b>：{guidelines['beginner_psychology']}
            </div>
        </div>
        """, unsafe_allow_html=True)

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

    # Sub-metrics
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

    # --- SECTION 3: VISUALIZATION CHART ---
    st.markdown("### 📈 高清回测图表分析 (Visual Performance Report)")
    if os.path.exists(chart_path):
        st.image(chart_path, caption=f"策略回测报表: {result.strategy_name} ({result.start_date} 至 {result.end_date})", use_container_width=True)
    else:
        st.warning("⚠️ 图表生成失败或文件不存在。")

    # --- SECTION 4: DATA TABLE & DOWNLOAD ---
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

        st.table(pd.DataFrame(table_rows))

        # Download daily equity CSV
        csv_data = result.equity_df.to_csv().encode("utf-8")
        st.download_button(
            label="📥 下载回测每日净值走势明细 (CSV)",
            data=csv_data,
            file_name=f"backtest_{selected_code}_{result.start_date}_{result.end_date}.csv",
            mime="text/csv",
        )
