"""
Module làm sạch dữ liệu và tính lợi suất
"""

import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from pathlib import Path
from typing import Tuple, Optional


def remove_weekends_holidays(df: pd.DataFrame, holidays_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loại bỏ thứ 7, chủ nhật và ngày lễ Việt Nam

    Args:
        df: DataFrame với cột 'date'
        holidays_path: Đường dẫn file holidays_vn.csv

    Returns:
        DataFrame đã lọc
    """
    df = df.copy()

    # Đảm bảo date là datetime
    if df['date'].dtype != 'datetime64[ns]':
        df['date'] = pd.to_datetime(df['date'])

    print(f"📅 Làm sạch dữ liệu...")
    print(f"   Ban đầu: {len(df)} records")

    # Loại bỏ thứ 7 (5) và chủ nhật (6)
    df = df[df['date'].dt.dayofweek < 5]
    print(f"   Sau khi bỏ T7/CN: {len(df)} records")

    # Loại bỏ ngày lễ
    if holidays_path and Path(holidays_path).exists():
        try:
            holidays_df = pd.read_csv(holidays_path)
            holidays = pd.to_datetime(holidays_df['date'])

            # Remove holidays
            df = df[~df['date'].isin(holidays)]
            print(f"   Sau khi bỏ ngày lễ: {len(df)} records ({len(holidays)} ngày lễ)")
        except Exception as e:
            print(f"   ⚠ Không đọc được file ngày lễ: {str(e)}")
    else:
        print(f"   ⚠ Không có file ngày lễ (bỏ qua)")

    return df


def calculate_returns(prices_df: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
    """
    Tính lợi suất từ giá đóng cửa

    Args:
        prices_df: DataFrame với cột date, ticker, close
        method: 'log' (log return) hoặc 'simple' (simple return)

    Returns:
        DataFrame với cột date, ticker, close, return
    """
    print(f"\n📈 Tính lợi suất ({method})...")

    df = prices_df.copy()

    # Sort by ticker and date
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    # Calculate returns for each ticker
    returns = []

    for ticker in df['ticker'].unique():
        df_ticker = df[df['ticker'] == ticker].copy()

        # Remove rows with zero or negative prices
        df_ticker = df_ticker[df_ticker['close'] > 0]

        if len(df_ticker) < 2:
            print(f"   ⚠ {ticker}: Không đủ dữ liệu (chỉ {len(df_ticker)} records)")
            continue

        if method == 'log':
            # Log return: ln(P_t / P_{t-1})
            df_ticker['return'] = np.log(df_ticker['close'] / df_ticker['close'].shift(1))
        else:
            # Simple return: (P_t - P_{t-1}) / P_{t-1}
            df_ticker['return'] = df_ticker['close'].pct_change()

        # Remove inf and -inf values
        df_ticker = df_ticker[np.isfinite(df_ticker['return'])]

        returns.append(df_ticker)

    df_returns = pd.concat(returns, ignore_index=True)

    # Drop first row (NaN return) for each ticker
    df_returns = df_returns.dropna(subset=['return'])

    print(f"   ✓ {df_returns['ticker'].nunique()} mã")
    print(f"   ✓ {len(df_returns)} returns")
    print(f"   ✓ Return range: {df_returns['return'].min():.4f} → {df_returns['return'].max():.4f}")

    return df_returns


def winsorize_returns(df: pd.DataFrame, limits: Tuple[float, float] = (0.01, 0.01)) -> pd.DataFrame:
    """
    Winsorize lợi suất để loại bỏ outliers

    Args:
        df: DataFrame với cột 'return'
        limits: Tuple (lower_percentile, upper_percentile)
                Ví dụ: (0.01, 0.01) = winsorize ở 1% và 99%

    Returns:
        DataFrame với returns đã winsorize
    """
    print(f"\n🔧 Winsorize returns tại [{limits[0]*100:.0f}%, {(1-limits[1])*100:.0f}%]...")

    df = df.copy()

    # Winsorize by ticker
    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        returns = df.loc[mask, 'return'].values

        if len(returns) > 0:
            # Winsorize
            returns_win = winsorize(returns, limits=limits)
            df.loc[mask, 'return'] = returns_win

    print(f"   ✓ Return range sau winsorize: {df['return'].min():.4f} → {df['return'].max():.4f}")

    return df


def detect_outliers(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """
    Phát hiện outliers bằng Z-score

    Args:
        df: DataFrame với cột 'return'
        threshold: Ngưỡng Z-score (mặc định 3.0)

    Returns:
        DataFrame với cột 'is_outlier' (boolean)
    """
    df = df.copy()

    # Calculate Z-score for each ticker
    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        returns = df.loc[mask, 'return']

        z_scores = np.abs((returns - returns.mean()) / returns.std())
        df.loc[mask, 'is_outlier'] = z_scores > threshold

    n_outliers = df['is_outlier'].sum()
    print(f"   ℹ Phát hiện {n_outliers} outliers (Z-score > {threshold})")

    return df


def clean_pipeline(prices_df: pd.DataFrame,
                   holidays_path: Optional[str] = 'config/holidays_vn.csv',
                   return_method: str = 'log',
                   winsorize_limits: Tuple[float, float] = (0.01, 0.01),
                   remove_outliers: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pipeline làm sạch dữ liệu đầy đủ

    Args:
        prices_df: DataFrame giá raw
        holidays_path: Đường dẫn file ngày lễ
        return_method: 'log' hoặc 'simple'
        winsorize_limits: Tuple (lower, upper) percentile
        remove_outliers: Có remove outliers không (dựa trên Z-score)

    Returns:
        Tuple (prices_clean, returns_clean)
    """
    print("\n" + "="*60)
    print("🧹 PIPELINE LÀM SẠCH DỮ LIỆU")
    print("="*60)

    # Step 1: Remove weekends & holidays
    df_clean = remove_weekends_holidays(prices_df, holidays_path)

    # Step 2: Calculate returns
    df_returns = calculate_returns(df_clean, method=return_method)

    # Step 3: Detect outliers (optional remove)
    if remove_outliers:
        df_returns = detect_outliers(df_returns, threshold=3.0)
        before = len(df_returns)
        df_returns = df_returns[~df_returns['is_outlier']]
        print(f"   ✓ Đã loại bỏ {before - len(df_returns)} outliers")

    # Step 4: Winsorize returns
    df_returns = winsorize_returns(df_returns, limits=winsorize_limits)

    # Summary
    print("\n" + "="*60)
    print("✅ HOÀN TẤT PIPELINE")
    print("="*60)
    print(f"   Prices: {len(df_clean)} records, {df_clean['ticker'].nunique()} mã")
    print(f"   Returns: {len(df_returns)} records")
    print(f"   Khoảng thời gian: {df_clean['date'].min()} → {df_clean['date'].max()}")

    return df_clean, df_returns


def get_return_statistics(df_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Tính các thống kê mô tả cho lợi suất

    Args:
        df_returns: DataFrame với cột ticker, return

    Returns:
        DataFrame thống kê
    """
    stats = df_returns.groupby('ticker')['return'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('25%', lambda x: x.quantile(0.25)),
        ('50%', 'median'),
        ('75%', lambda x: x.quantile(0.75)),
        ('max', 'max'),
        ('skew', lambda x: x.skew()),
        ('kurt', lambda x: x.kurtosis())
    ]).round(6)

    return stats


def check_stationarity_summary(df_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Kiểm tra tính dừng của chuỗi lợi suất (ADF test preview)

    Note: Full ADF test sẽ được thực hiện trong module ARIMA
    Đây chỉ là preview nhanh

    Returns:
        DataFrame với summary
    """
    from statsmodels.tsa.stattools import adfuller

    results = []

    for ticker in df_returns['ticker'].unique():
        returns = df_returns[df_returns['ticker'] == ticker]['return'].dropna()

        if len(returns) > 10:
            try:
                adf_result = adfuller(returns, maxlag=1)
                results.append({
                    'ticker': ticker,
                    'adf_stat': adf_result[0],
                    'p_value': adf_result[1],
                    'is_stationary': adf_result[1] < 0.05
                })
            except:
                results.append({
                    'ticker': ticker,
                    'adf_stat': np.nan,
                    'p_value': np.nan,
                    'is_stationary': False
                })

    df_results = pd.DataFrame(results)

    print("\n📊 Kiểm tra tính dừng (ADF test):")
    print(f"   {df_results['is_stationary'].sum()}/{len(df_results)} chuỗi dừng (p < 0.05)")

    return df_results


def export_cleaned_data(prices_clean: pd.DataFrame,
                       returns_clean: pd.DataFrame,
                       output_dir: str = 'data_cache') -> dict:
    """
    Xuất dữ liệu đã làm sạch

    Returns:
        Dict với paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Export to parquet (efficient)
    prices_path = output_path / 'prices_clean.parquet'
    returns_path = output_path / 'returns_clean.parquet'

    prices_clean.to_parquet(prices_path, index=False)
    returns_clean.to_parquet(returns_path, index=False)

    print(f"\n💾 Đã xuất dữ liệu:")
    print(f"   Prices: {prices_path}")
    print(f"   Returns: {returns_path}")

    return {
        'prices': str(prices_path),
        'returns': str(returns_path)
    }