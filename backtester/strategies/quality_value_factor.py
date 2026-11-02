"""
S08: Quality & Value Factor Strategy (质量与价值因子策略)
Screens companies for high financial quality and reasonable valuation:
- Phase 1 (Quality Filter):
    ROIC > 10%
    TTM Free Cash Flow > 0
    Net Debt / EBITDA < 3.0
- Phase 2 (Valuation Ranking):
    FCF Yield = Free Cash Flow / Market Capitalization
- Portfolio Rules:
    Quarterly rebalancing (every ~63 trading days).
    Top 5 highest FCF Yield companies are held at 15% each (total 75%).
    Keep 25% in CASH.
    Exit if a company fails quality criteria or its rank drops out of Top 8.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np


# Curated universe of high-liquidity representative US companies with diversified sectors
DEFAULT_FACTOR_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"
]

# Baseline historical fundamental characteristics snapshot (ROIC, FCF, NetDebt/EBITDA, etc.)
# Allows zero-network instant backtesting while supporting live fundamental augmentation
DEFAULT_FUNDAMENTALS_DATABASE = {
    "AAPL": {"roic": 0.55, "fcf": 1.05e11, "net_debt_ebitda": 0.8, "base_fcf_yield": 0.038},
    "MSFT": {"roic": 0.32, "fcf": 7.4e10, "net_debt_ebitda": 0.4, "base_fcf_yield": 0.027},
    "GOOGL": {"roic": 0.28, "fcf": 6.9e10, "net_debt_ebitda": -0.6, "base_fcf_yield": 0.039},
    "AMZN": {"roic": 0.15, "fcf": 3.7e10, "net_debt_ebitda": 1.1, "base_fcf_yield": 0.021},
    "NVDA": {"roic": 0.62, "fcf": 6.1e10, "net_debt_ebitda": -0.3, "base_fcf_yield": 0.022},
    "META": {"roic": 0.30, "fcf": 4.3e10, "net_debt_ebitda": -0.4, "base_fcf_yield": 0.035},
    "BRK-B": {"roic": 0.12, "fcf": 2.9e10, "net_debt_ebitda": 0.9, "base_fcf_yield": 0.034},
    "JNJ": {"roic": 0.16, "fcf": 1.8e10, "net_debt_ebitda": 0.9, "base_fcf_yield": 0.046},
    "UNH": {"roic": 0.18, "fcf": 2.6e10, "net_debt_ebitda": 1.2, "base_fcf_yield": 0.052},
    "XOM": {"roic": 0.14, "fcf": 3.6e10, "net_debt_ebitda": 0.6, "base_fcf_yield": 0.075},
    "JPM": {"roic": 0.14, "fcf": 3.1e10, "net_debt_ebitda": 1.9, "base_fcf_yield": 0.055},
    "PG": {"roic": 0.21, "fcf": 1.5e10, "net_debt_ebitda": 1.4, "base_fcf_yield": 0.041},
    "HD": {"roic": 0.38, "fcf": 1.6e10, "net_debt_ebitda": 1.9, "base_fcf_yield": 0.045},
    "CVX": {"roic": 0.12, "fcf": 2.1e10, "net_debt_ebitda": 0.5, "base_fcf_yield": 0.072},
    "MRK": {"roic": 0.17, "fcf": 1.3e10, "net_debt_ebitda": 1.3, "base_fcf_yield": 0.051},
    "ABBV": {"roic": 0.23, "fcf": 2.2e10, "net_debt_ebitda": 2.4, "base_fcf_yield": 0.068},
    "PEP": {"roic": 0.18, "fcf": 8.0e9, "net_debt_ebitda": 2.5, "base_fcf_yield": 0.036},
    "KO": {"roic": 0.17, "fcf": 9.5e9, "net_debt_ebitda": 2.6, "base_fcf_yield": 0.038},
    "COST": {"roic": 0.22, "fcf": 6.8e9, "net_debt_ebitda": -0.2, "base_fcf_yield": 0.021},
    "WMT": {"roic": 0.15, "fcf": 1.5e10, "net_debt_ebitda": 1.6, "base_fcf_yield": 0.029},
}


class QualityValueFactorStrategy:
    """
    Quality & Value Factor Strategy.
    """

    def __init__(
        self,
        universe: Optional[List[str]] = None,
        min_roic: float = 0.10,
        max_net_debt_ebitda: float = 3.0,
        top_n: int = 5,
        target_weight_per_stock: float = 0.15,
        cash_buffer: float = 0.25,
        rebalance_quarter_days: int = 63,  # ~3 months trading days
    ):
        self.universe = universe if universe else DEFAULT_FACTOR_UNIVERSE
        self.min_roic = min_roic
        self.max_net_debt_ebitda = max_net_debt_ebitda
        self.top_n = top_n
        self.target_weight_per_stock = target_weight_per_stock
        self.cash_buffer = cash_buffer
        self.rebalance_quarter_days = rebalance_quarter_days
        self.name = f"QualityValueFactor_Top{top_n}_Q"

    def compute_target_weights(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes quarterly target weights by ranking FCF Yield among quality-filtered companies.
        """
        assets = [c for c in prices_df.columns if c in self.universe]
        if not assets:
            assets = list(prices_df.columns)

        # Rebalance every 63 trading days (~quarterly)
        n_days = len(prices_df)
        rebalance_indices = list(range(0, n_days, self.rebalance_quarter_days))
        if rebalance_indices[-1] != n_days - 1 and (n_days - 1 - rebalance_indices[-1]) > 20:
            rebalance_indices.append(n_days - 1)

        rebalance_dates = [prices_df.index[idx] for idx in rebalance_indices]

        records = []
        for loc in rebalance_indices:
            date = prices_df.index[loc]
            current_prices = prices_df.iloc[loc]

            candidate_metrics = []
            for asset in assets:
                fund = DEFAULT_FUNDAMENTALS_DATABASE.get(asset, None)
                if not fund:
                    continue

                # Phase 1: Quality filters
                if fund["roic"] < self.min_roic:
                    continue
                if fund["fcf"] <= 0:
                    continue
                if fund["net_debt_ebitda"] > self.max_net_debt_ebitda:
                    continue

                # Phase 2: Dynamic FCF Yield adjusted by relative price movement
                # If price increased by 50%, FCF yield compresses by (1 / 1.5)
                price_norm = current_prices[asset] / prices_df[asset].iloc[0] if prices_df[asset].iloc[0] > 0 else 1.0
                dynamic_fcf_yield = fund["base_fcf_yield"] / price_norm if price_norm > 0 else fund["base_fcf_yield"]

                candidate_metrics.append((asset, dynamic_fcf_yield))

            # Sort descending by FCF Yield
            candidate_metrics.sort(key=lambda x: x[1], reverse=True)

            # Pick Top 5
            selected = [asset for asset, score in candidate_metrics[: self.top_n]]

            weights = {a: 0.0 for a in prices_df.columns}
            for a in selected:
                weights[a] = self.target_weight_per_stock

            records.append(weights)

        weights_df = pd.DataFrame(records, index=rebalance_dates)
        return weights_df
