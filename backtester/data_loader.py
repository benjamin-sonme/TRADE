"""
Data Loader Module
Fetches and caches historical market data using yfinance.
"""

import os
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


class DataLoader:
    """Handles downloading and caching historical OHLCV data."""

    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir

    def get_cache_path(self, ticker: str, start: str, end: str) -> str:
        safe_ticker = ticker.replace("^", "INDEX_").upper()
        return os.path.join(self.cache_dir, f"{safe_ticker}_{start}_{end}.csv")

    def load_data(
        self,
        ticker: str,
        start_date: str = None,
        end_date: str = None,
        lookback_years: int = 3,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Load historical OHLCV data for a ticker.
        If start_date is not provided, defaults to lookback_years before end_date.
        """
        ticker = ticker.strip().upper()

        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")

        if start_date is None:
            start_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=lookback_years * 365)
            start_date = start_dt.strftime("%Y-%m-%d")

        cache_path = self.get_cache_path(ticker, start_date, end_date)

        if not force_refresh:
            # 1. Check exact cache match
            if os.path.exists(cache_path):
                try:
                    df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                    print(f"[DataLoader] Loaded cached data for {ticker} ({len(df)} bars from {start_date} to {end_date})")
                    return df
                except Exception as e:
                    print(f"[DataLoader] Failed to read exact cache ({e}), checking alternatives...")

            # 2. Check if any cache exists for this ticker (useful for recently IPO'd stocks)
            prefix = f"{ticker}_"
            for fname in os.listdir(self.cache_dir):
                if fname.startswith(prefix) and fname.endswith(".csv"):
                    try:
                        df = pd.read_csv(os.path.join(self.cache_dir, fname), index_col=0, parse_dates=True)
                        if len(df) > 0:
                            print(f"[DataLoader] Found existing local cache '{fname}' ({len(df)} bars) for {ticker}. Using cached data.")
                            return df
                    except Exception:
                        pass

        print(f"[DataLoader] Downloading historical data for {ticker} from {start_date} to {end_date}...")
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download data for '{ticker}' via yfinance: {e}\n"
                f"Tip: Check ticker spelling, network connection, or run with network permission."
            )

        if df is None or df.empty:
            raise ValueError(
                f"No data returned for ticker '{ticker}' between {start_date} and {end_date}. "
                f"Please verify if the symbol is correct and public on US exchanges."
            )

        # Handle multi-index columns if returned by yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure standard OHLCV column naming
        df.columns = [col.capitalize() for col in df.columns]

        # Ensure datetime index without timezone
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df.dropna(subset=["Close"])
        df.sort_index(inplace=True)

        # Save to cache
        df.to_csv(cache_path)

        print(f"[DataLoader] Downloaded {len(df)} trading days for {ticker}.")
        return df
