# U.S. Equity Trading Strategy Backtesting System (美股交易策略回测系统 v1.0)

专为美股交易初学者设计的策略回测与决策辅助系统。用于验证交易策略在真实历史行情中的表现，克服主观交易中的“犹豫不决”与“冲动下单”，建立基于数据和纪律的交易认知。

---

## 🌟 系统核心功能 (v1.0)

1. **自动获取并缓存历史行情数据**：
   - 支持美股任意公开标的（如 `NVDA`, `TSLA`, `AAPL`, `SPY` 等）。
   - 自动在本地 `data_cache/` 进行离线缓存，避免重复网络下载，秒级运行。
2. **两套经典交易策略（完全零未来函数 / 严格无前视偏差）**：
   - **双均线交叉策略 (Dual MA Crossover)**：金叉买入、死叉卖出。支持 SMA/EMA，可自由调整短期与长期窗口。
   - **RSI 均值回归策略 (RSI Mean Reversion)**：RSI 超卖逢低买入、超买止盈，内嵌长期趋势过滤器（如 200 日线）防止抄底跌入深渊。
3. **专业回测指标全量计算**：
   - **累计收益率 (Cumulative Return)**
   - **年化收益率 (CAGR)** 与 **年化波动率 (Volatility)**
   - **夏普比率 (Sharpe Ratio)**：以年化无风险利率（默认 4.0%）计算每单位波动风险获取的超额回报。
   - **最大回撤 (Max Drawdown, MDD)** 与 **最长回撤恢复天数**
   - **胜率 (Win Rate)**、**盈亏比 (Profit Factor)** 与 **交易次数**
4. **基准对比 (Benchmark Comparison)**：
   - 每一笔策略回测均与该标的的 **“买入并持有 (Buy & Hold)”** 策略进行全方位对照。
5. **专业可视化图表输出**：
   - 自动生成 3 栏对比图并保存至 `reports/`：
     - 上栏：策略净值曲线 vs 买入持有基准曲线。
     - 中栏：标的股价走势、技术指标及买入（绿色箭头）/ 卖出（红色箭头）点位标记。
     - 下栏：水下回撤图（Underwater Drawdown），直观感受回撤深度与防守效果。

---

## 🚀 快速上手

在终端处于该项目根目录时：

### 1. 交互式向导模式（推荐初学者）
直接运行：
```bash
.venv/bin/python main.py
```
系统会一步步提示你输入股票代码（如 `NVDA`）、选择策略（均线或 RSI）、选择回测周期，并自动运行输出报告。

### 2. 命令行快捷运行
```bash
# 测试英伟达 (NVDA) 最近 3 年的双均线策略
.venv/bin/python main.py --ticker NVDA --strategy ma --years 3

# 测试特斯拉 (TSLA) 最近 3 年的 RSI 策略
.venv/bin/python main.py --ticker TSLA --strategy rsi --years 3

# 自定义均线参数 (例如 10日线 金叉 30日线)
.venv/bin/python main.py --ticker TSLA --strategy ma --short-ma 10 --long-ma 30
```

---

## 📌 标的特别说明（关于 SpaceX、特斯拉与英伟达）

* **SpaceX (纳斯达克代码: `SPCX`)**：
  * SpaceX 已正式在纳斯达克挂牌上市，股票代码为 **`SPCX`**。
  * 你在系统中无论是输入 `SPCX` 还是直接输入 `SPACEX`，系统都会自动识别并获取上市以来的真实日 K 线数据进行回测。
* **特斯拉 (TSLA) & 英伟达 (NVDA)**：
  * 美股流动性极佳的公开上市公司，数据完备，系统已完成历史数据离线缓存。
* **衍生关联标的**：
  * **DXYZ (Destiny Tech100)**：持有包括 SpaceX 在内多家头部独角兽私募份额的公开上市封闭基金，系统内同样已支持回测。

---

## 📂 项目结构

```
TRADE/
├── backtester/
│   ├── data_loader.py       # 历史数据加载与本地缓存
│   ├── strategies/
│   │   ├── base.py          # 策略基类抽象接口
│   │   ├── ma_crossover.py  # 双均线金叉死叉策略
│   │   └── rsi_strategy.py  # RSI 超买超卖与趋势过滤策略
│   ├── engine.py            # 回测执行引擎（含滑点、手续费与交易跟踪）
│   ├── metrics.py           # 收益、夏普比率、最大回撤等指标计算
│   └── visualizer.py        # 3 栏专业对比图表绘制
├── data_cache/              # 历史行情本地缓存目录
├── reports/                 # 生成的可视化分析图表
├── main.py                  # 交互式 CLI 与命令行运行入口
└── requirements.txt         # Python 依赖清单
```
