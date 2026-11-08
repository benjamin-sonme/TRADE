"""
Risk Analyzer Module
Evaluates investment risk levels (R1 to R4), risk factors (drawdown, volatility, Sharpe),
and generates actionable risk management advice for investors and trading beginners.
"""

from typing import Dict, Any


def assess_investment_risk(metrics: Dict[str, Any], ticker: str = "TICKER") -> Dict[str, Any]:
    """
    Computes an investment risk profile and rating (R1 - R4) based on
    Max Drawdown, Volatility, Sharpe Ratio, and benchmark comparisons.
    """
    mdd = abs(metrics.get("Strategy Max Drawdown", 0.0))
    vol = abs(metrics.get("Strategy Volatility", 0.0))
    sharpe = metrics.get("Strategy Sharpe", 0.0)
    ret = metrics.get("Strategy Total Return", 0.0)

    # Risk Scoring Algorithm (0 - 100, higher score = higher risk)
    # 1. Drawdown score (up to 45 pts)
    # 10% MDD -> 9 pts, 30% MDD -> 27 pts, 50% MDD -> 45 pts
    dd_score = min(45.0, (mdd / 0.50) * 45.0)

    # 2. Volatility score (up to 40 pts)
    # 20% vol -> 16 pts, 40% vol -> 32 pts, 50%+ vol -> 40 pts
    vol_score = min(40.0, (vol / 0.50) * 40.0)

    # 3. Sharpe penalty/discount (up to 15 pts)
    # If Sharpe < 0.5 -> high penalty; if Sharpe > 1.2 -> discount
    if sharpe < 0:
        sharpe_score = 15.0
    elif sharpe < 0.5:
        sharpe_score = 10.0
    elif sharpe < 1.0:
        sharpe_score = 5.0
    else:
        sharpe_score = 0.0

    total_risk_score = dd_score + vol_score + sharpe_score

    # Determine Risk Tier (R1 ~ R4)
    if total_risk_score <= 35 and mdd <= 0.18:
        level = "R1 (低风险 - 稳健防御型)"
        icon = "🟢"
        badge = "LOW_RISK"
        max_position = "25% - 30%"
        stop_loss = "4% - 6%"
        style = "适合作为投资组合的核心底仓，资产波动极小，回撤防御极佳。"
        psychology = "持仓心态极其安稳，无需频繁盯盘，适合中长期定投持有。"
    elif total_risk_score <= 55 and mdd <= 0.28:
        level = "R2 (中低风险 - 平衡稳健型)"
        icon = "🟡"
        badge = "MODERATE_RISK"
        max_position = "15% - 20%"
        stop_loss = "6% - 8%"
        style = "收益与波动平衡，兼顾适度成长性与防守纪律。"
        psychology = "能承受温和的短期浮动回撤，波段持有体验良好。"
    elif total_risk_score <= 75 and mdd <= 0.45:
        level = "R3 (中高风险 - 进取成长型)"
        icon = "🟠"
        badge = "HIGH_RISK"
        max_position = "10% - 15%"
        stop_loss = "7% - 10%"
        style = "典型高贝塔科技成长龙头（如英伟达、特斯拉），具有爆发力但波段剧烈。"
        psychology = "必须做好账面短期浮亏 20%~30% 的心理准备，切忌满仓一把梭，严禁追高！"
    else:
        level = "R4 (极高风险 - 激进投机型)"
        icon = "🔴"
        badge = "EXTREME_RISK"
        max_position = "5% - 8% (轻仓试探)"
        stop_loss = "5% (严格硬止损)"
        style = "高估值暴涨暴跌或次新高波动资产，极易发生腰斩级别深套。"
        psychology = "必须严格执行机械止损纪律，绝不可抱有侥幸心理逆势抗单！"

    return {
        "ticker": ticker,
        "risk_level": level,
        "icon": icon,
        "badge": badge,
        "risk_score": total_risk_score,
        "max_drawdown": mdd,
        "volatility": vol,
        "sharpe_ratio": sharpe,
        "max_position": max_position,
        "stop_loss": stop_loss,
        "style": style,
        "psychology": psychology,
    }


def print_risk_assessment_card(risk_profile: Dict[str, Any]):
    """Prints a structured Investment Risk Rating card in terminal."""
    icon = risk_profile["icon"]
    level = risk_profile["risk_level"]
    ticker = risk_profile["ticker"]

    print("\n" + "=" * 75)
    print(f" 🛡️ 【{ticker}】投资风险评级与初学者风控指南 (Risk Assessment Card)")
    print("=" * 75)
    print(f"  {icon} 综合风险评级：【{level}】 (风险综合得分: {risk_profile['risk_score']:.1f}/100)")
    print("  ---------------------------------------------------------------------")
    print(f"  📌 核心风控三要素体检：")
    print(f"     • 历史最大回撤 (MDD)  : {risk_profile['max_drawdown']*100:.2f}%  (承受深套风险)")
    print(f"     • 年化波动剧烈度 (Vol) : {risk_profile['volatility']*100:.2f}%  (日常颠簸剧烈程度)")
    print(f"     • 收益风险比 (Sharpe) : {risk_profile['sharpe_ratio']:.2f}     (每承担一分风险获取的超额回报)")
    print("  ---------------------------------------------------------------------")
    print(f"  🎯 初学者操作风控红线 (Disciplined Guidelines)：")
    print(f"     1. 建议单股仓位上限 : 建议占总资金不超过 【{risk_profile['max_position']}】")
    print(f"     2. 建议防守止损线   : 跌破进场价 【{risk_profile['stop_loss']}】 必须坚决触发离场")
    print(f"     3. 资产适配定位     : {risk_profile['style']}")
    print(f"     4. 交易心理建设     : {risk_profile['psychology']}")
    print("=" * 75 + "\n")
