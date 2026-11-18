# U.S. Equity Trading Strategy Backtesting & Signal Preview System (美股量化回测与信号预览系统 v2.3)

专为美股交易者设计的**策略回测、资产配置与前瞻性信号分析系统**。不仅用于验证交易策略在真实历史行情中的表现，还通过 **Next-Day Signal Preview（下一交易日信号预览）** 精确回答“根据当前已知信息，下一交易日必须发生什么条件才会触发新买卖信号”，帮助克服主观交易中的“犹豫不决”与“冲动下单”。

---

## 🌟 最新版本 v2.3 核心升级 (Key Updates)

### 1. 🔮 Next-Day Signal Preview（下一交易日信号预览系统 / Forward Signal Analysis）
将系统从纯粹的历史回测工具升级为辅助实际决策的前瞻性分析平台，严格遵守 **“单一真相源（Single Source of Truth）”**——历史回测与下一日信号预览调用完全一致的策略逻辑，杜绝双套条件漂移与未来函数。

- **双层级信号体系（杜绝未来函数）**：
  - **Confirmed Signal（已确认信号）**：基于最新已完成收盘日（`T`）数据锁定，若触发新拐点，最早可于下一交易日（`T+1`）执行。
  - **Conditional Signal（条件性信号）**：精确求解下一交易日（`T+1`）需满足的价格/状态条件；若 `T+1` 收盘触发，须于 `T+1` 盘后确认，最早于 `T+2` 执行。
- **五项核心决策信息**：
  - `Current Signal`（当前信号）、`Next Buy Trigger`（下一日买入触发条件）、`Next Sell Trigger`（下一日卖出触发条件）、`Distance to Trigger`（距离触发价百分比与进度条）、`Execution Timing`（最早执行时点），并附 **`Why This Signal?`** 指标计算明细。
- **9 大策略三分级建模**：
  - **精确/状态条件反推（S01 / S02 / S03 / S06）**：严格处理状态前置条件（如 S02 的 `RSI ≤ 30` 与 `SMA200` 趋势线交集、S06 连续 2 日跌破 `SMA50` 及近 5 日回踩状态），条件互斥或未就绪时明确显示“当前无法触发”，绝不强行编造价格。
  - **动态指标与多资产情景（S05 / S04 / S07 / S09）**：S05 动态模拟下一日收盘对布林带 2σ 上轨的影响；S04 / S07 / S09 输出多资产排名、容忍带偏移表与下一调仓检验日。
  - **基本面边界约束（S08）**：明确标注财报与估值主导边界，展示因子横截面明细与季频调仓倒计时，**不编造虚假的纯股价触发点**。
- **风控与执行校验（Risk & Execution Validator）**：
  - 联动 R1~R4 投资风险评级模型，自动校验模拟持仓状态、现金充足度/底线、高风险标的仓位红线及防未来函数执行时点。
- **两大深度验证与推演工具**：
  - **🎚️ 价格敏感性分析（Price Sensitivity）**：支持自定义假设明日收盘价（如 `160, 170, 180, 190`），实时查看单策略动态指标曲线（ECharts）及 **`S01 / S02 / S03 / S05 / S06` 跨策略信号横向对比矩阵**。
  - **🧪 历史重放验证（Historical Replay Validation）**：将历史每一天 `T` 当作“今天”并严格隐藏 `T+1` 及之后所有数据，验证信号预览模块与原始回测引擎的一致率（防未来函数与逻辑漂移审计）。

### 2. 🕯️ 交互式 ECharts K 线与诊断视图（v2.2 增强）
- **实时行情同步与系统代理自动识别**：自动检测 macOS/系统代理环境抓取 Yahoo Finance 最新美股交易日数据（并保留离线本地缓存容错回退）。
- **触控板双指缩放 + 底部滑动条 + 全屏展开**：支持在 K 线图与策略诊断图上双指缩放、拖拽底部滑块精细浏览局部波段。
- **交易次数诊断与逻辑拆解**：针对 0 交易或少交易场景自动剖析拦截原因（如 S02 的 200 日线拦截、期末未平仓统计口径说明）。
- **一键返回首页**：Web 端支持从回测报告或信号预览页一键复位回初始主界面。

### 3. 📊 基础回测核心能力（v2.1 延续）
- **7 档时间周期 + 指标充分预热**：支持 `5y / 3y / 1y / 6m / 1m / 1w / 3d`，超短周期先用完整历史预热均线/RSI/布林带后再无偏差切片。
- **纳斯达克市值前 50 强标的池**：内置 AAPL、NVDA、MSFT、AMZN、GOOGL、META、TSLA、AVGO、COST、ASML、SPCX 等 50 家龙头标的。
- **投资风险评级卡 (R1~R4)**：基于最大回撤、年化波动率与夏普比率生成综合评分与初学者仓位/止损风控红线。

---

## 🌟 完整 9 大经典交易策略矩阵 (S01 - S09)

系统现已完整内置从**单股技术择时**到**多资产配置轮动**的 9 大策略：

| 编号 | 策略名称 | 所属流派 | 核心逻辑与交易规则 | 下一日触发计算类别 |
| :--- | :--- | :--- | :--- | :--- |
| **S01** | **Dual MA Crossover** | 趋势跟踪 | 20 SMA 金叉 50 SMA 买入，死叉清仓避险 | 数学反推精确阈值 |
| **S02** | **RSI Mean Reversion** | 均值回归 | RSI ≤ 30 超卖且位于 200 日线上买入，RSI ≥ 70 止盈 | 递推公式反推阈值（含趋势交集） |
| **S03** | **Donchian Breakout** | 突破交易 | 突破过去 20 日最高价买入，跌破过去 10 日最低价卖出（海龟收盘确认版） | 历史已知阈值 |
| **S04** | **Relative Momentum** | 相对动量 | 每月末按 $M_t = \frac{P_{t-21}}{P_{t-252}} - 1$ 排序候选池，Top 4 正动量各配 20%，保留 ≥20% 现金 | 多资产排名与调仓日情景 |
| **S05** | **Bollinger Band Squeeze** | 波动率突破 | 带宽处于过去 126 日最低 10% 挤压区，5 日内向上突破上轨买入，跌破 20 日中轨卖出 | 动态指标模型模拟 |
| **S06** | **Trend Pullback** | 趋势回撤 | 50日>200日线；5日内最低价触及50日线；今日收盘重回20日线上买入；连续两日跌破50日线卖出 | 均线与状态条件反推 |
| **S07** | **Threshold Rebalancing** | 组合再平衡 | 容忍带 $Threshold_i = \max(2\%, 25\% \times TW_i)$，月末检查超标止盈、低配补仓，守住现金底线 | 组合权重容忍带与调仓日 |
| **S08** | **Quality & Value Factor** | 基本面因子 | 季频筛选 ROIC ≥ 10%、FCF > 0、Net Debt/EBITDA ≤ 3，按 FCF Yield 取 Top 5（各 15%，留 25% 现金） | 基本面财报主导（非纯价格触发） |
| **S09** | **Dual Momentum** | 全球股债轮动 | 月末比较 12 个月收益：SPY vs VEU 选胜者，胜者跑赢 BIL 则全仓持有，否则 100% 切入 BIL 避险 | 多 ETF 相对/绝对动量情景 |

---

## 🚀 快速上手与运行

### 1. 现代化 Web 可视化控制台（推荐，支持在 VS Code 终端一键启动）
在终端或 VS Code 内置终端中执行以下任意一条命令，浏览器会自动打开 `http://localhost:8501`：
```bash
./run_ui.sh
# 或
.venv/bin/streamlit run app.py
# 或
.venv/bin/python main.py --ui
```
- **工作区双模式自由切换**：
  - **📊 历史回测分析**：风险评级卡 (R1~R4)、核心收益指标、交易触发原因诊断、ECharts 诊断 K 线、全景回测报告（支持即时预览与按需下载 PNG / 导出 CSV），并在结果顶部嵌入下一日信号速览卡。
  - **🔮 下一交易日信号预览 (Next-Day Signal)**：双层级信号卡（Confirmed vs Conditional）、无截断 2 列核心触发价卡片、距离触发进度条、`Why This Signal?` 指标拆解、风控校验器、自定义价格敏感性矩阵与历史无前视重放验证台。

---

### 2. 终端交互式向导与命令行直测
```bash
# 交互式向导模式
.venv/bin/python main.py

# 单股技术指标策略
.venv/bin/python main.py --strategy s01 --ticker NVDA
.venv/bin/python main.py --strategy s03 --ticker TSLA -p 6m
.venv/bin/python main.py --strategy s05 --ticker NVDA -p 1m --no-chart
.venv/bin/python main.py --strategy s06 --ticker SPCX -p 3d

# 多资产组合与纳指 50 强轮动策略
.venv/bin/python main.py --strategy s04 --nasdaq-top50 -p 6m
.venv/bin/python main.py --strategy s07 -p 1y
.venv/bin/python main.py --strategy s08 --nasdaq-top50
.venv/bin/python main.py --strategy s09
```

---

## 📌 标的特别说明（关于 SpaceX、特斯拉与英伟达）

* **SpaceX (纳斯达克代码: `SPCX`)**：支持直接输入 `SPCX` 或 `SPACEX`，系统自动识别并获取真实日 K 线数据。
* **特斯拉 (`TSLA`) & 英伟达 (`NVDA`)**：核心科技龙头，数据完备并具备本地缓存与实时刷新同步能力。
* **衍生关联标的**：支持 `DXYZ (Destiny Tech100)` 等美股公开上市代码自定义输入。

---

## 📂 项目架构

```
TRADE/
├── backtester/
│   ├── data_loader.py               # 单股及多资产行情加载、自动代理检测、实时刷新与本地缓存
│   ├── engine.py                    # 单资产回测引擎（严格 T+1 滞后、滑点、手续费、周期切片）
│   ├── portfolio_engine.py          # 多资产组合与动态再平衡回测引擎
│   ├── universe.py                  # 纳斯达克市值前 50 强标的池映射
│   ├── risk_analyzer.py             # 投资风险评级模型 (R1~R4) 与初学者风控指南
│   ├── metrics.py                   # 收益率、CAGR、夏普比率、最大回撤、波动率等指标
│   ├── trade_diagnosis.py           # 交易触发次数与拦截原因诊断引擎
│   ├── chart_ui.py                  # ECharts 交互式 K 线与策略诊断图构建器（支持缩放与滑块）
│   ├── visualizer.py                # 单股全景回测图表渲染（支持内存 PNG 与本地保存）
│   ├── portfolio_visualizer.py      # 组合净值与动态资产配置堆叠图渲染
│   ├── forward_signal_ui.py         # 下一交易日信号预览 Streamlit 交互仪表盘组件
│   ├── forward_signal/              # 下一交易日信号预览核心引擎（Forward Signal Analysis）
│   │   ├── __init__.py              # 统一入口 analyze_next_day_signal 与导出接口
│   │   ├── models.py                # TriggerCondition / ForwardSignalReport 等数据模型
│   │   ├── scenario_engine.py       # 单一真相源策略工厂与 ForwardScenarioEngine
│   │   ├── threshold_solver.py      # 单股策略精确/状态/动态临界阈值求解器 (S01/S02/S03/S05/S06)
│   │   ├── portfolio_solver.py      # 多资产情景、容忍带与基本面边界求解器 (S04/S07/S08/S09)
│   │   ├── risk_validator.py        # 风控与防未来函数执行校验器 (Risk & Execution Validator)
│   │   └── analysis_tools.py        # 价格敏感性分析（含跨策略矩阵）与无前视历史重放验证
│   └── strategies/
│       ├── base.py                  # 策略抽象基类
│       ├── ma_crossover.py          # [S01] 双均线交叉策略
│       ├── rsi_strategy.py          # [S02] RSI 均值回归策略
│       ├── donchian_breakout.py     # [S03] 唐奇安通道突破策略
│       ├── relative_momentum.py     # [S04] 相对动量轮动策略
│       ├── bollinger_squeeze.py     # [S05] 布林带挤压突破策略
│       ├── trend_pullback.py        # [S06] 趋势回撤买入策略
│       ├── threshold_rebalancing.py # [S07] 阈值再平衡策略
│       ├── quality_value_factor.py  # [S08] 质量与价值因子策略
│       └── dual_momentum.py         # [S09] 双重动量资产轮动策略
├── data_cache/                      # 历史行情本地缓存目录
├── reports/                         # 用户按需保存的回测图表目录
├── app.py                           # Streamlit 交互式 Web UI 主程序 (v2.3)
├── run_ui.sh                        # 一键启动 Web UI 脚本
├── main.py                          # 交互式 CLI 与命令行入口
└── requirements.txt                 # 依赖清单
```
