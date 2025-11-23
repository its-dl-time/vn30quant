"""
Smart Data Manager - Quản lý cache thông minh
Tự động quyết định load từ cache hay gọi API

Logic:
- IF cache exists AND cache_date >= (today - 1):
    → Load cache (fast)
- ELSE:
    → Fetch API + save cache

- Returns: Nếu cache có cùng date range → load cache
          Ngược lại: tính lại + save cache
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, List
import json

# Import existing modules
from .data_io import smart_fetch, fetch_vnindex
from .clean import clean_pipeline


class DataManager:
    """
    Smart cache manager cho VN30 data
    """

    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Metadata file
        self.metadata_file = self.cache_dir / "cache_metadata.json"


    # ========================================================================
    # METADATA MANAGEMENT
    # ========================================================================

    def _load_metadata(self) -> Dict:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}


    def _save_metadata(self, metadata: Dict):
        """Save cache metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)


    def _update_metadata(self, key: str, value: any):
        """Update single metadata entry"""
        meta = self._load_metadata()
        meta[key] = value
        self._save_metadata(meta)


    # ========================================================================
    # CACHE VALIDATION
    # ========================================================================

    def _is_cache_fresh(self, cache_date: str, max_age_days: int = 1) -> bool:
        """
        Kiểm tra cache có fresh không

        Args:
            cache_date: Ngày cuối cùng trong cache (YYYY-MM-DD)
            max_age_days: Cache tối đa bao nhiêu ngày (default=1)

        Returns:
            True nếu cache đủ mới
        """
        try:
            cache_dt = pd.to_datetime(cache_date).date()
            today = datetime.now().date()

            # Cache fresh nếu: cache_date >= (today - max_age_days)
            threshold = today - timedelta(days=max_age_days)

            return cache_dt >= threshold
        except:
            return False


    def _check_cache_exists(self, cache_prefix: str = "prices") -> Tuple[bool, Optional[str]]:
        """
        Kiểm tra cache có tồn tại và lấy last_date

        Returns:
            (exists, last_date)
        """
        cache_file = self.cache_dir / f"{cache_prefix}_cache.parquet"

        if not cache_file.exists():
            return False, None

        try:
            df = pd.read_parquet(cache_file)
            if 'date' in df.columns and len(df) > 0:
                last_date = pd.to_datetime(df['date']).max().strftime('%Y-%m-%d')
                return True, last_date
            else:
                return False, None
        except:
            return False, None


    # ========================================================================
    # SMART LOAD - PRICES
    # ========================================================================

    def load_prices(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
        max_cache_age_days: int = 1
    ) -> pd.DataFrame:
        """
        Smart load prices với cache

        Args:
            tickers: List mã cổ phiếu
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD
            force_refresh: Bỏ qua cache, tải mới
            max_cache_age_days: Cache tối đa bao nhiêu ngày (default=1)

        Returns:
            DataFrame prices
        """
        print("\n" + "="*60)
        print("📦 SMART LOAD PRICES")
        print("="*60)

        cache_file = self.cache_dir / "prices_cache.parquet"

        # Check cache
        cache_exists, cache_last_date = self._check_cache_exists("prices")

        use_cache = False

        if not force_refresh and cache_exists:
            # Validate cache freshness
            if self._is_cache_fresh(cache_last_date, max_cache_age_days):
                print(f"✅ Cache tìm thấy (last_date: {cache_last_date})")
                print(f"   Cache đủ mới (< {max_cache_age_days} ngày)")

                # Load cache và filter theo tickers + date range
                try:
                    df_cache = pd.read_parquet(cache_file)
                    df_cache['date'] = pd.to_datetime(df_cache['date'])

                    # Filter
                    df_filtered = df_cache[
                        (df_cache['ticker'].isin(tickers)) &
                        (df_cache['date'] >= pd.to_datetime(start_date)) &
                        (df_cache['date'] <= pd.to_datetime(end_date))
                    ]

                    # Check if cache covers requested range
                    if len(df_filtered) > 0:
                        cache_tickers = set(df_filtered['ticker'].unique())
                        requested_tickers = set(tickers)

                        if requested_tickers.issubset(cache_tickers):
                            print(f"   ✓ Cache có đủ {len(requested_tickers)} tickers")
                            print(f"   ✓ Load {len(df_filtered)} records từ cache")
                            use_cache = True
                            return df_filtered
                        else:
                            missing = requested_tickers - cache_tickers
                            print(f"   ⚠ Cache thiếu {len(missing)} tickers: {missing}")
                    else:
                        print(f"   ⚠ Cache không có data trong range yêu cầu")

                except Exception as e:
                    print(f"   ⚠ Lỗi đọc cache: {str(e)}")
            else:
                print(f"⏰ Cache cũ (last_date: {cache_last_date})")
                print(f"   Cần refresh (> {max_cache_age_days} ngày)")

        # Fetch from API
        if not use_cache:
            print(f"\n📥 Tải dữ liệu từ API...")
            print(f"   Tickers: {tickers}")
            print(f"   Range: {start_date} → {end_date}")

            df_new = smart_fetch(tickers, start_date, end_date, source='auto')

            # Save to cache
            df_new.to_parquet(cache_file, index=False)
            print(f"\n💾 Đã lưu cache: {cache_file}")

            # Update metadata
            self._update_metadata('prices_last_update', datetime.now().isoformat())
            self._update_metadata('prices_last_date', df_new['date'].max())
            self._update_metadata('prices_tickers', list(df_new['ticker'].unique()))

            return df_new


    # ========================================================================
    # SMART LOAD - VNINDEX
    # ========================================================================

    def load_vnindex(
        self,
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
        max_cache_age_days: int = 1
    ) -> pd.DataFrame:
        """
        Smart load VNINDEX với cache
        """
        print("\n" + "="*60)
        print("📊 SMART LOAD VNINDEX")
        print("="*60)

        cache_file = self.cache_dir / "vnindex_cache.parquet"

        # Check cache
        cache_exists, cache_last_date = self._check_cache_exists("vnindex")

        if not force_refresh and cache_exists:
            if self._is_cache_fresh(cache_last_date, max_cache_age_days):
                print(f"✅ Cache VNINDEX (last_date: {cache_last_date})")

                try:
                    df_cache = pd.read_parquet(cache_file)
                    df_cache['date'] = pd.to_datetime(df_cache['date'])

                    # Filter date range
                    df_filtered = df_cache[
                        (df_cache['date'] >= pd.to_datetime(start_date)) &
                        (df_cache['date'] <= pd.to_datetime(end_date))
                    ]

                    if len(df_filtered) > 0:
                        print(f"   ✓ Load {len(df_filtered)} records từ cache")
                        return df_filtered

                except Exception as e:
                    print(f"   ⚠ Lỗi đọc cache: {str(e)}")
            else:
                print(f"⏰ Cache VNINDEX cũ (last_date: {cache_last_date})")

        # Fetch from API
        print(f"\n📥 Tải VNINDEX từ API...")
        df_new = fetch_vnindex(start_date, end_date, source='auto')

        # Save cache
        df_new.to_parquet(cache_file, index=False)
        print(f"\n💾 Đã lưu cache: {cache_file}")

        self._update_metadata('vnindex_last_update', datetime.now().isoformat())
        self._update_metadata('vnindex_last_date', df_new['date'].max())

        return df_new


    # ========================================================================
    # SMART LOAD - RETURNS (with auto clean)
    # ========================================================================

    def load_returns(
        self,
        prices_df: pd.DataFrame,
        force_recalculate: bool = False,
        holidays_path: str = 'config/holidays_vn.csv',
        return_method: str = 'log',
        winsorize_limits: Tuple[float, float] = (0.01, 0.01)
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Smart load returns với cache

        Args:
            prices_df: DataFrame prices (đã load)
            force_recalculate: Bắt buộc tính lại
            holidays_path: Path ngày lễ
            return_method: 'log' hoặc 'simple'
            winsorize_limits: Winsorize limits

        Returns:
            (prices_clean, returns_clean)
        """
        print("\n" + "="*60)
        print("📈 SMART LOAD RETURNS")
        print("="*60)

        cache_prices_file = self.cache_dir / "prices_clean_cache.parquet"
        cache_returns_file = self.cache_dir / "returns_cache.parquet"

        # Get date range of input prices
        prices_start = prices_df['date'].min()
        prices_end = prices_df['date'].max()

        # Check if returns cache matches prices
        use_cache = False

        if not force_recalculate and cache_returns_file.exists():
            try:
                df_returns_cache = pd.read_parquet(cache_returns_file)
                df_returns_cache['date'] = pd.to_datetime(df_returns_cache['date'])

                cache_start = df_returns_cache['date'].min()
                cache_end = df_returns_cache['date'].max()

                # Check if cache covers same range
                if cache_start <= prices_start and cache_end >= prices_end:
                    cache_tickers = set(df_returns_cache['ticker'].unique())
                    prices_tickers = set(prices_df['ticker'].unique())

                    if prices_tickers.issubset(cache_tickers):
                        print(f"✅ Returns cache tìm thấy")
                        print(f"   Cache range: {cache_start.date()} → {cache_end.date()}")
                        print(f"   Prices range: {prices_start.date()} → {prices_end.date()}")
                        print(f"   ✓ Cache phủ đủ range → Load cache")

                        # Load prices_clean cache
                        if cache_prices_file.exists():
                            df_prices_clean = pd.read_parquet(cache_prices_file)
                            df_prices_clean['date'] = pd.to_datetime(df_prices_clean['date'])
                        else:
                            df_prices_clean = prices_df  # Fallback

                        # Filter to match input range
                        df_returns_filtered = df_returns_cache[
                            (df_returns_cache['ticker'].isin(prices_tickers)) &
                            (df_returns_cache['date'] >= prices_start) &
                            (df_returns_cache['date'] <= prices_end)
                        ]

                        df_prices_filtered = df_prices_clean[
                            (df_prices_clean['ticker'].isin(prices_tickers)) &
                            (df_prices_clean['date'] >= prices_start) &
                            (df_prices_clean['date'] <= prices_end)
                        ]

                        print(f"   ✓ Load {len(df_returns_filtered)} returns từ cache")

                        use_cache = True
                        return df_prices_filtered, df_returns_filtered
                    else:
                        print(f"   ⚠ Cache thiếu một số tickers")
                else:
                    print(f"   ⚠ Cache không phủ đủ date range")

            except Exception as e:
                print(f"   ⚠ Lỗi đọc returns cache: {str(e)}")

        # Calculate returns
        if not use_cache:
            print(f"\n🧹 Tính returns mới (clean pipeline)...")

            prices_clean, returns_clean = clean_pipeline(
                prices_df,
                holidays_path=holidays_path,
                return_method=return_method,
                winsorize_limits=winsorize_limits,
                remove_outliers=False
            )

            # Save cache
            prices_clean.to_parquet(cache_prices_file, index=False)
            returns_clean.to_parquet(cache_returns_file, index=False)

            print(f"\n💾 Đã lưu returns cache:")
            print(f"   {cache_prices_file}")
            print(f"   {cache_returns_file}")

            # Update metadata
            self._update_metadata('returns_last_update', datetime.now().isoformat())
            self._update_metadata('returns_date_range', {
                'start': returns_clean['date'].min(),
                'end': returns_clean['date'].max()
            })

            return prices_clean, returns_clean


    # ========================================================================
    # UTILITIES
    # ========================================================================

    def get_cache_info(self) -> Dict:
        """Get cache information"""
        meta = self._load_metadata()

        info = {
            'metadata': meta,
            'files': {}
        }

        # Check cache files
        for cache_type in ['prices', 'vnindex', 'prices_clean', 'returns']:
            cache_file = self.cache_dir / f"{cache_type}_cache.parquet"

            if cache_file.exists():
                try:
                    df = pd.read_parquet(cache_file)
                    info['files'][cache_type] = {
                        'exists': True,
                        'size_kb': cache_file.stat().st_size / 1024,
                        'rows': len(df),
                        'columns': list(df.columns)
                    }

                    if 'date' in df.columns:
                        info['files'][cache_type]['date_range'] = {
                            'start': df['date'].min(),
                            'end': df['date'].max()
                        }
                except:
                    info['files'][cache_type] = {'exists': True, 'error': 'Cannot read'}
            else:
                info['files'][cache_type] = {'exists': False}

        return info


    def clear_cache(self, cache_type: Optional[str] = None):
        """
        Clear cache

        Args:
            cache_type: 'prices', 'vnindex', 'returns', 'all', or None (all)
        """
        if cache_type is None or cache_type == 'all':
            # Clear all
            for f in self.cache_dir.glob("*_cache.parquet"):
                f.unlink()
                print(f"🗑️ Deleted: {f}")

            if self.metadata_file.exists():
                self.metadata_file.unlink()
                print(f"🗑️ Deleted: {self.metadata_file}")
        else:
            # Clear specific
            cache_file = self.cache_dir / f"{cache_type}_cache.parquet"
            if cache_file.exists():
                cache_file.unlink()
                print(f"🗑️ Deleted: {cache_file}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def smart_load_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    load_vnindex: bool = True,
    cache_dir: str = "data_cache",
    max_cache_age_days: int = 1,
    force_refresh: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function: Load tất cả data cần thiết

    Returns:
        {
            'prices': DataFrame,
            'returns': DataFrame,
            'prices_clean': DataFrame,
            'vnindex': DataFrame (nếu load_vnindex=True)
        }
    """
    manager = DataManager(cache_dir)

    result = {}

    # 1. Load prices
    prices = manager.load_prices(
        tickers, start_date, end_date,
        force_refresh=force_refresh,
        max_cache_age_days=max_cache_age_days
    )
    result['prices'] = prices

    # 2. Load VNINDEX
    if load_vnindex:
        vnindex = manager.load_vnindex(
            start_date, end_date,
            force_refresh=force_refresh,
            max_cache_age_days=max_cache_age_days
        )
        result['vnindex'] = vnindex

    # 3. Load returns (with clean)
    prices_clean, returns = manager.load_returns(prices, force_recalculate=force_refresh)
    result['prices_clean'] = prices_clean
    result['returns'] = returns

    return result