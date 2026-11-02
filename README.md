# U.S. Equity Trading Strategy Backtesting System (美股交易策略回测系统 v2.0)

专为美股交易者设计的策略回测、资产配置与决策辅助系统。用于验证交易策略在真实历史行情中的表现，克服主观交易中的“犹豫不决”与“冲动下单”，建立基于数据和纪律的量化认知。

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

### 1. 交互式向导模式（推荐）
在终端或 VS Code 中运行：
```bash
.venv/bin/python main.py
```
终端会弹出 1-9 号策略列表，跟随引导即可一键运行任何单股或投资组合策略！

### 2. 命令行直测指令速查

#### 【单股技术指标策略】
```bash
# S01: 测试 TSLA 双均线策略
.venv/bin/python main.py --strategy s01 --ticker TSLA

# S02: 测试 NVDA 的 RSI 均值回归策略
.venv/bin/python main.py --strategy s02 --ticker NVDA

# S03: 测试 TSLA 的唐奇安通道突破策略
.venv/bin/python main.py --strategy s03 --ticker TSLA

# S05: 测试 NVDA 的布林带挤压突破策略
.venv/bin/python main.py --strategy s05 --ticker NVDA

# S06: 测试 TSLA 的趋势回撤买入策略
.venv/bin/python main.py --strategy s06 --ticker TSLA
```

#### 【多资产组合与宏观轮动策略】
```bash
# S04: 测试候选股票池的相对动量月度轮动策略
.venv/bin/python main.py --strategy s04

# S07: 测试持仓组合的月度阈值动态再平衡策略
.venv/bin/python main.py --strategy s07

# S08: 测试财报高质量与自由现金流价值因子策略
.venv/bin/python main.py --strategy s08

# S09: 测试 Gary Antonacci 经典的全球股债双重动量轮动策略 (SPY/VEU/BIL)
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
│   ├── engine.py                 # 单资产回测引擎（严格无前视偏差、滑点、手续费）
│   ├── portfolio_engine.py       # 多资产组合与动态再平衡回测引擎
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
├── main.py                       # 交互式 CLI 与命令行入口
└── requirements.txt              # 依赖清单
```
