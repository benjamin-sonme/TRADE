# U.S. Equity Trading Strategy Backtesting System (美股交易策略回测系统 v2.1)

专为美股交易者设计的策略回测、资产配置与决策辅助系统。用于验证交易策略在真实历史行情中的表现，克服主观交易中的“犹豫不决”与“冲动下单”，建立基于数据和纪律的量化认知。

---

## 🌟 新版本 v2.1 核心特性 (Key Updates)

1. **多档位回测周期支持 (7 档时间周期)**：
   - 现已支持 **5年 (5y)**、**3年 (3y)**、**1年 (1y)**、**半年 (6m)**、**1个月 (1m)**、**1周 (1w)**、**3天 (3d)**。
   - **底线指标充分预热 (Warm-up Protection)**：超短周期（如 3 天、1 周）回测先预载充分的历史数据计算技术指标（如 200 日线、RSI、布林带），再在目标窗口进行无偏差切片仿真，杜绝因样本过短导致的指标 NaN 失效。
2. **纳斯达克市值前 50 强标的池 (Nasdaq Top 50 Universe)**：
   - 内置涵盖 AAPL、NVDA、MSFT、AMZN、GOOGL、META、TSLA、AVGO、COST、ASML、SPCX 等 50 家龙头标的。
   - 单股策略输入 `?` 或 `top50` 可一览完整清单；多资产策略（S04 动量轮动、S07 再平衡、S08 因子策略）可一键切换为纳指 50 强池。
3. **企业与策略投资风险评级卡 (Investment Risk Rating R1~R4)**：
   - 每次回测自动生成包含综合风险得分、风险等级与初学者风控红线的评价报告：
     - 🟢 **R1 (低风险 - 稳健防御型)**：回撤极小、防御性极佳，建议仓位 25%~30%，适合作为核心底仓。
     - 🟡 **R2 (中低风险 - 平衡稳健型)**：回撤可控、收益与风险平衡，建议仓位 15%~20%，适合波段配置。
     - 🟠 **R3 (中高风险 - 进取成长型)**：波动较大、成长性强，建议单股仓位 8%~12%，严格执行 8%~10% 止损。
     - 🔴 **R4 (极高风险 - 激进投机型)**：高波动、大回撤，单股仓位不得超过 5%，必须设硬止损。
4. **图表生成授权交互 (Interactive Chart Confirmation)**：
   - 回测完成后会在终端询问是否生成并保存图表 `(y/n) [默认 y]`，避免每次自动弹图与重复生成磁盘文件。
   - 命令行支持 `--chart`（静默生成）与 `--no-chart`（静默跳过）自动化参数。

---

## 🌟 完整 9 大经典交易策略矩阵 (S01 - S09)

系统现已完整内置从**单股技术择时**到**多资产配置轮动**的 9 大策略：

| 编号 | 策略名称 | 所属流派 | 核心逻辑与交易规则 | 适用标的 |
| :--- | :--- | :--- | :--- | :--- |
| **S01** | **Dual MA Crossover** | 趋势跟踪 | 20 SMA 金叉 50 SMA 买入，死叉清仓避险 | 单股 (如 NVDA/TSLA) |
| **S02** | **RSI Mean Reversion** | 均值回归 | RSI < 30 超卖且位于 200 日线上买入，RSI > 70 止盈 | 单股 |
| **S03** | **Donchian Breakout** | 突破交易 | 突破过去 20 日最高价买入，跌破过去 10 日最低价卖出（海龟交易法收盘确认版） | 单股 |
| **S04** | **Relative Momentum** | 相对动量 | 每月按 $M_t = \frac{P_{t-21}}{P_{t-252}} - 1$ 排序候选池，Top 4 且动量为正各配 20%，保留至少 20% 现金 | 股票池 (如科技/蓝筹) |
| **S05** | **Bollinger Band Squeeze** | 波动率突破 | 带宽处于过去 126 日最低 10% 挤压区，5 日内向上突破上轨买入，跌破 20 日均线卖出 | 单股 |
| **S06** | **Trend Pullback** | 趋势回撤 | 50日>200日线确认大趋势；5日内最低价触及50日线；今日收盘重新站上20日线买入；连续两日收盘跌破50日线卖出 | 单股 |
| **S07** | **Threshold Rebalancing**| 组合再平衡 | 设定目标仓位与容忍带 $Threshold_i = \max(2\%, 25\% \times TW_i)$，月度检查超标止盈、低配补仓，控制现金底线 | 多资产组合 |
| **S08** | **Quality & Value Factor**| 基本面因子 | 季频筛选：ROIC > 10%, FCF > 0, Net Debt/EBITDA < 3，按自由现金流收益率 (FCF Yield) 排序买入 Top 5，各 15%，保留 25% 现金 | 蓝筹基本面池 |
| **S09** | **Dual Momentum** | 全球股债轮动 | 月度计算过去 12 个月收益率：SPY vs VEU 选胜者，若胜者跑赢 BIL（短期国债）则持有胜者，否则 100% 持有 BIL 避险 | 宏观大类资产 |

---

## 🚀 快速上手与运行

### 1. 现代化 Web 可视化控制台（免终端操作，强烈推荐！）
在终端或 VS Code 中执行以下任意一条命令，系统将自动唤起浏览器打开交互式控制台：
```bash
./run_ui.sh
# 或
.venv/bin/python main.py --ui
```
- **功能特性**：
  - **左侧控制面板**：自由切换 9 大策略、从纳斯达克 50 强下拉秒选标的、切换 7 档时间跨度（3天~5年）、调节初始资金与滑点。
  - **右侧主仪表盘**：
    - 🛡️ **投资风险评级卡 (R1~R4)**：醒目的风险等级色彩徽章、综合风险评分进度条、仓位上限与止损防守红线提示。
    - 📊 **核心指标大卡片**：累计收益、年化复合收益率 (CAGR)、夏普比率、最大回撤等并排呈现。
    - 📈 **高清图表看板**：净值走势 vs 基准、买卖信号点 K 线图、水下回撤、动态资产配置堆叠图一目了然。
    - 📥 **数据明细与导出**：一键导出回测每日明细为 CSV 文件。

---

### 2. 终端交互式向导模式
在终端或 VS Code 中运行：
```bash
.venv/bin/python main.py
```
终端会弹出 1-9 号策略列表，跟随引导即可一键运行任何单股或投资组合策略！

### 2. 命令行直测指令速查

#### 【单股技术指标策略】
```bash
# 测试 NVDA 双均线策略 (默认3年)
.venv/bin/python main.py --strategy s01 --ticker NVDA

# 测试 TSLA 唐奇安突破策略 (半年期，自动询问是否绘图)
.venv/bin/python main.py --strategy s03 --ticker TSLA -p 6m

# 测试 NVDA 布林带挤压策略 (1个月期，跳过绘图)
.venv/bin/python main.py --strategy s05 --ticker NVDA -p 1m --no-chart

# 测试 SPCX (SpaceX) 趋势回撤策略 (3天超短周期)
.venv/bin/python main.py --strategy s06 --ticker SPCX -p 3d
```

#### 【多资产组合与纳指50强轮动策略】
```bash
# S04: 相对动量轮动 (纳斯达克 Top 50 股票池，半年期)
.venv/bin/python main.py --strategy s04 --nasdaq-top50 -p 6m

# S07: 阈值动态再平衡 (1年期)
.venv/bin/python main.py --strategy s07 -p 1y

# S08: 财报高质量与价值因子策略 (纳斯达克 Top 50 股票池)
.venv/bin/python main.py --strategy s08 --nasdaq-top50

# S09: Gary Antonacci 全球股债双重动量轮动 (SPY/VEU/BIL)
.venv/bin/python main.py --strategy s09
```

---

## 📊 专业可视化输出与回测指标

每次运行策略，系统不仅打印详细的指标对照表，还在 `reports/` 目录自动生成高清图表：
1. **单股策略图表**：
   - 上栏：策略净值 vs 标的买入持有 (Buy & Hold)
   - 中栏：股价 K 线、技术指标（均线、通道、布林带）及精准的买入（绿色▲）与卖出（红色▼）标记
   - 下栏：水下回撤曲线（Underwater Drawdown）
2. **多资产组合图表**：
   - 上栏：组合净值走势 vs 宏观基准
   - 中栏：**动态资产配置堆叠图 (Stacked Allocation)**，清晰展现各阶段不同股票与现金的持仓百分比变化
   - 下栏：组合最大回撤曲线

---

## 📌 标的特别说明（关于 SpaceX、特斯拉与英伟达）

* **SpaceX (纳斯达克代码: `SPCX`)**：
  * SpaceX 已正式在纳斯达克挂牌上市，股票代码为 **`SPCX`**。
  * 你在系统中无论是输入 `SPCX` 还是直接输入 `SPACEX`，系统都会自动识别并获取真实日 K 线数据进行回测。
* **特斯拉 (TSLA) & 英伟达 (NVDA)**：
  * 美股流动性极佳的公开上市公司，数据完备，系统已完成历史数据离线缓存。
* **衍生关联标的**：
  * **DXYZ (Destiny Tech100)**：持有包括 SpaceX 在内多家头部独角兽私募份额的公开上市封闭基金，系统内同样已支持回测。

---

## 📂 项目架构

```
TRADE/
├── backtester/
│   ├── data_loader.py            # 单股及多资产历史数据加载与本地缓存
│   ├── engine.py                 # 单资产回测引擎（严格无前视偏差、滑点、手续费、周期切片）
│   ├── portfolio_engine.py       # 多资产组合与动态再平衡回测引擎
│   ├── universe.py               # 纳斯达克市值前 50 强标的池映射与展示
│   ├── risk_analyzer.py          # 投资风险评级模型 (R1~R4) 与初学者风控指南
│   ├── metrics.py                # 收益率、夏普比率、最大回撤、波动率等量化指标
│   ├── visualizer.py             # 单股买卖信号与水下回撤绘图
│   ├── portfolio_visualizer.py   # 组合净值与动态资产配置堆积图绘制
│   └── strategies/
│       ├── base.py               # 策略抽象基类
│       ├── ma_crossover.py       # [S01] 双均线交叉策略
│       ├── rsi_strategy.py       # [S02] RSI 均值回归策略
│       ├── donchian_breakout.py  # [S03] 唐奇安通道突破策略
│       ├── relative_momentum.py  # [S04] 相对动量轮动策略
│       ├── bollinger_squeeze.py  # [S05] 布林带挤压突破策略
│       ├── trend_pullback.py     # [S06] 趋势回撤买入策略
│       ├── threshold_rebalancing.py # [S07] 阈值再平衡策略
│       ├── quality_value_factor.py  # [S08] 质量与价值因子策略
│       └── dual_momentum.py      # [S09] 双重动量资产轮动策略
├── data_cache/                   # 历史行情本地缓存目录
├── reports/                      # 策略回测生成的图表文件
├── app.py                        # Streamlit 交互式 Web UI 控制台
├── run_ui.sh                     # 一键启动 Web UI 脚本
├── main.py                       # 交互式 CLI 与命令行入口
└── requirements.txt              # 依赖清单
```
