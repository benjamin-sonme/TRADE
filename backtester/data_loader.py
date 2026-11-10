"""
Data Loader Module
Fetches and caches historical market data using yfinance.
"""

import os
import time
import urllib.request
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
NY_TZ = ZoneInfo("America/New_York")


def _configure_system_proxy():
    """Reads OS proxy settings (e.g., macOS System Proxy) so curl_cffi / yfinance can connect cleanly."""
    try:
        proxies = urllib.request.getproxies()
        if proxies.get("https") and not os.environ.get("HTTPS_PROXY"):
            os.environ["HTTPS_PROXY"] = proxies["https"]
            os.environ["https_proxy"] = proxies["https"]
        if proxies.get("http") and not os.environ.get("HTTP_PROXY"):
            os.environ["HTTP_PROXY"] = proxies["http"]
            os.environ["http_proxy"] = proxies["http"]
    except Exception:
        pass


def _expected_us_market_date(ny_now: datetime) -> date:
    """Returns the most recent US equity trading session date based on NY market clock."""
    wd = ny_now.weekday()
    if wd == 5:
        return ny_now.date() - timedelta(days=1)
    if wd == 6:
        return ny_now.date() - timedelta(days=2)
    # Weekday before 09:30 AM ET -> previous trading weekday
    if (ny_now.hour, ny_now.minute) < (9, 30):
        return ny_now.date() - timedelta(days=3 if wd == 0 else 1)
    return ny_now.date()


def _is_us_market_open(ny_now: datetime) -> bool:
    """Checks if US equity regular session is currently open (09:30 - 16:20 ET, Mon-Fri)."""
    if ny_now.weekday() >= 5:
        return False
    hm = (ny_now.hour, ny_now.minute)
    return (9, 30) <= hm <= (16, 20)


class DataLoader:
    """Handles downloading and caching historical & latest OHLCV market data."""

    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        _configure_system_proxy()

    def get_cache_path(self, ticker: str, start: str, end: str) -> str:
        safe_ticker = ticker.replace("^", "INDEX_").upper()
        return os.path.join(self.cache_dir, f"{safe_ticker}_{start}_{end}.csv")

    def _is_cache_fresh(self, fpath: str, df: pd.DataFrame, req_start_ts: pd.Timestamp, ny_now: datetime, is_live_today: bool) -> bool:
        """Checks whether a cached CSV covers both the start lookback and the latest US session."""
        if df.empty:
            return False

        covers_start = df.index.min() <= req_start_ts + pd.Timedelta(days=20)
        if not covers_start:
            # Handle recent IPOs (e.g., SPCX) whose filename already queried from req_start_ts or earlier
            parts = os.path.basename(fpath).replace(".csv", "").split("_")
            if len(parts) >= 3:
                try:
                    if pd.Timestamp(parts[-2]) <= req_start_ts + pd.Timedelta(days=5):
                        covers_start = True
                except Exception:
                    pass
        if not covers_start:
            return False

        last_bar_date = pd.Timestamp(df.index.max()).date()
        expected_date = _expected_us_market_date(ny_now)
        mtime_age = time.time() - os.path.getmtime(fpath)

        # If asking up to today/live:
        if is_live_today:
            if last_bar_date < expected_date:
                # Could be a US holiday (e.g. MLK, Good Friday) if already fetched within 15 min
                if mtime_age < 900 and os.path.basename(fpath).endswith(f"_{ny_now.strftime('%Y-%m-%d')}.csv"):
                    return True
                return False
            # If market is live right now, refresh every 5 minutes
            if _is_us_market_open(ny_now) and mtime_age > 300:
                return False
        return True

    def load_data(
        self,
        ticker: str,
        start_date: str = None,
        end_date: str = None,
        lookback_years: int = 3,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Load historical & latest OHLCV data for a ticker.
        Automatically refreshes stale caches so charts always capture the latest US trading day.
        """
        ticker = ticker.strip().upper()
        ny_now = datetime.now(NY_TZ)
        is_live_today = end_date is None

        if end_date is None:
            end_date = ny_now.strftime("%Y-%m-%d")

        if start_date is None:
            start_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=lookback_years * 365)
            start_date = start_dt.strftime("%Y-%m-%d")

        # yfinance treats `end` as exclusive; add 2 days when downloading up to today
        yf_end_date = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=2)).strftime("%Y-%m-%d")
        cache_path = self.get_cache_path(ticker, start_date, end_date)
        req_start_ts = pd.Timestamp(start_date)

        fallback_df = None
        if not force_refresh:
            # 1. Check exact cache match
            if os.path.exists(cache_path):
                try:
                    df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                    if self._is_cache_fresh(cache_path, df, req_start_ts, ny_now, is_live_today):
                        print(f"[DataLoader] Loaded fresh cache for {ticker} ({len(df)} bars, latest {df.index.max().strftime('%Y-%m-%d')})")
                        return df
                    if len(df) > 0:
                        fallback_df = df
                except Exception as e:
                    print(f"[DataLoader] Failed to read exact cache ({e}), checking alternatives...")

            # 2. Check if another cache exists that covers both start_date and latest US session
            prefix = f"{ticker}_"
            for fname in sorted(os.listdir(self.cache_dir), reverse=True):
                if fname.startswith(prefix) and fname.endswith(".csv"):
                    fpath = os.path.join(self.cache_dir, fname)
                    try:
                        df = pd.read_csv(fpath, index_col=0, parse_dates=True)
                        if len(df) > 0:
                            if self._is_cache_fresh(fpath, df, req_start_ts, ny_now, is_live_today):
                                print(f"[DataLoader] Found fresh local cache '{fname}' ({len(df)} bars, latest {df.index.max().strftime('%Y-%m-%d')}) for {ticker}.")
                                return df
                            if fallback_df is None or df.index.max() > fallback_df.index.max() or (
                                df.index.max() == fallback_df.index.max() and len(df) > len(fallback_df)
                            ):
                                fallback_df = df
                    except Exception:
                        pass

        print(f"[DataLoader] Downloading latest historical data for {ticker} from {start_date} to {end_date}...")
        try:
            df = yf.download(ticker, start=start_date, end=yf_end_date, progress=False, auto_adjust=True)
        except Exception as e:
            if fallback_df is not None:
                print(f"[DataLoader] Download failed ({e}), falling back to cached data ({len(fallback_df)} bars, latest {fallback_df.index.max().strftime('%Y-%m-%d')}) for {ticker}.")
                return fallback_df
            raise RuntimeError(
                f"Failed to download data for '{ticker}' via yfinance: {e}\n"
                f"Tip: Check ticker spelling, network connection, or run with network permission."
            )

        if df is None or df.empty:
            if fallback_df is not None:
                print(f"[DataLoader] Empty download, falling back to cached data ({len(fallback_df)} bars) for {ticker}.")
                return fallback_df
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

        print(f"[DataLoader] Downloaded {len(df)} trading days for {ticker} (latest {df.index.max().strftime('%Y-%m-%d')}).")
        return df

    def load_multi_data(
        self,
        tickers: list,
        start_date: str = None,
        end_date: str = None,
        lookback_years: int = 3,
        price_col: str = "Close",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Loads multiple tickers and returns a unified DataFrame of Close prices.
        """
        price_dict = {}
        for t in tickers:
            t = t.strip().upper()
            if t in ["SPACEX", "SPACE-X", "SPACE_X"]:
                t = "SPCX"
            try:
                df = self.load_data(
                    t,
                    start_date=start_date,
                    end_date=end_date,
                    lookback_years=lookback_years,
                    force_refresh=force_refresh,
                )
                price_dict[t] = df[price_col]
            except Exception as e:
                print(f"[DataLoader] Warning: Could not load data for {t}: {e}")

        if not price_dict:
            raise ValueError(f"Failed to load data for any tickers in {tickers}")

        combined_df = pd.DataFrame(price_dict)
        # Drop rows where ALL tickers are NaN
        combined_df.dropna(how="all", inplace=True)
        combined_df.ffill(inplace=True)
        return combined_df
