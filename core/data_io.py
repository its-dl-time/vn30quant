"""
Module lấy dữ liệu giá từ CafeF API / VNDirect / SSI hoặc CSV
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def get_session_with_retries():
    """Tạo session với retry logic"""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def fetch_prices_cafef(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Lấy dữ liệu từ CafeF API (chính)

    Args:
        tickers: Danh sách mã chứng khoán
        start: Ngày bắt đầu (YYYY-MM-DD)
        end: Ngày kết thúc (YYYY-MM-DD)

    Returns:
        DataFrame với cột: date, ticker, open, high, low, close, volume
    """
    all_data = []

    # Convert dates
    start_dt = datetime.strptime(start, '%Y-%m-%d')
    end_dt = datetime.strptime(end, '%Y-%m-%d')

    print(f"📥 Đang tải từ CafeF API...")
    print(f"   Thời gian: {start} → {end}")

    session = get_session_with_retries()

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'vi,en-US;q=0.9,en;q=0.8',
        'Referer': 'https://s.cafef.vn/',
        'Origin': 'https://s.cafef.vn'
    }

    for ticker in tickers:
        try:
            # CafeF API endpoint
            url = "https://s.cafef.vn/Ajax/PageNew/DataHistory/PriceHistory.ashx"

            params = {
                'Symbol': ticker,
                'StartDate': start_dt.strftime('%m/%d/%Y'),
                'EndDate': end_dt.strftime('%m/%d/%Y'),
                'PageIndex': 1,
                'PageSize': 10000
            }

            response = session.get(url, params=params, headers=headers, timeout=15)
            response.raise_for_status()

            data = response.json()

            if 'Data' in data and 'Data' in data['Data'] and len(data['Data']['Data']) > 0:
                records = data['Data']['Data']

                df_temp = pd.DataFrame(records)

                # Map columns - CafeF format
                df_temp = df_temp.rename(columns={
                    'Ngay': 'date',
                    'GiaDongCua': 'close',
                    'GiaMoCua': 'open',
                    'GiaCaoNhat': 'high',
                    'GiaThapNhat': 'low',
                    'KhoiLuongKhopLenh': 'volume'
                })

                df_temp['ticker'] = ticker.upper()

                # Convert date string to datetime
                df_temp['date'] = pd.to_datetime(df_temp['date'], format='%d/%m/%Y')

                # CRITICAL FIX: Filter to requested date range
                df_temp = df_temp[
                    (df_temp['date'] >= pd.to_datetime(start_dt)) &
                    (df_temp['date'] <= pd.to_datetime(end_dt))
                ]

                # Select columns
                cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
                df_temp = df_temp[cols]

                # Convert to numeric
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')

                all_data.append(df_temp)
                print(f"   ✓ {ticker}: {len(df_temp)} records")
            else:
                print(f"   ⚠ {ticker}: Không có dữ liệu")

            time.sleep(0.3)

        except requests.exceptions.RequestException as e:
            print(f"   ✗ {ticker}: Lỗi kết nối - {str(e)[:60]}")
        except KeyError as e:
            print(f"   ✗ {ticker}: Lỗi format - {str(e)}")
        except Exception as e:
            print(f"   ✗ {ticker}: {str(e)[:60]}")

    if not all_data:
        raise ValueError("Không lấy được dữ liệu từ CafeF. Vui lòng dùng CSV!")

    df = pd.concat(all_data, ignore_index=True)
    df = unify_prices(df)

    print(f"\n✅ Hoàn tất: {len(df)} records từ {df['ticker'].nunique()} mã")
    return df


def fetch_prices_vnd(tickers: List[str], start: str, end: str, timeout: int = 10) -> pd.DataFrame:
    """
    Fallback: VNDirect API
    """
    all_data = []

    start_date = datetime.strptime(start, '%Y-%m-%d').strftime('%Y-%m-%d')
    end_date = datetime.strptime(end, '%Y-%m-%d').strftime('%Y-%m-%d')

    print(f"📥 Đang tải từ VNDirect API...")

    session = get_session_with_retries()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
    }

    for ticker in tickers:
        try:
            url = "https://finfo-api.vndirect.com.vn/v4/stock_prices"
            query = f"code:{ticker}~fromDate:gte:{start_date}~toDate:lte:{end_date}"
            params = {'sort': 'date', 'size': 9999, 'page': 1, 'q': query}

            response = session.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()

            data = response.json()

            if 'data' in data and len(data['data']) > 0:
                df_temp = pd.DataFrame(data['data'])
                df_temp = df_temp.rename(columns={
                    'code': 'ticker', 'date': 'date',
                    'open': 'open', 'high': 'high', 'low': 'low',
                    'close': 'close', 'nmVolume': 'volume'
                })
                cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
                df_temp = df_temp[cols]
                all_data.append(df_temp)
                print(f"   ✓ {ticker}: {len(df_temp)} records")
            else:
                print(f"   ⚠ {ticker}: Không có dữ liệu")

            time.sleep(0.3)

        except Exception as e:
            print(f"   ✗ {ticker}: {str(e)[:60]}")

    if not all_data:
        raise ValueError("Không lấy được dữ liệu từ VNDirect!")

    df = pd.concat(all_data, ignore_index=True)
    df = unify_prices(df)

    print(f"\n✅ Hoàn tất: {len(df)} records từ {df['ticker'].nunique()} mã")
    return df


def fetch_prices_ssi(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Fallback: SSI iBoard API
    """
    all_data = []

    print(f"📥 Đang tải từ SSI API...")

    session = get_session_with_retries()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
    }

    start_ts = int(datetime.strptime(start, '%Y-%m-%d').timestamp())
    end_ts = int(datetime.strptime(end, '%Y-%m-%d').timestamp())

    for ticker in tickers:
        try:
            url = "https://iboard.ssi.com.vn/dchart/api/history"
            params = {
                'resolution': 'D',
                'symbol': ticker,
                'from': start_ts,
                'to': end_ts
            }

            response = session.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data.get('s') == 'ok' and len(data.get('t', [])) > 0:
                df_temp = pd.DataFrame({
                    'date': pd.to_datetime(data['t'], unit='s'),
                    'ticker': ticker,
                    'open': data['o'],
                    'high': data['h'],
                    'low': data['l'],
                    'close': data['c'],
                    'volume': data['v']
                })
                all_data.append(df_temp)
                print(f"   ✓ {ticker}: {len(df_temp)} records")
            else:
                print(f"   ⚠ {ticker}: Không có dữ liệu")

            time.sleep(0.2)

        except Exception as e:
            print(f"   ✗ {ticker}: {str(e)[:60]}")

    if not all_data:
        raise ValueError("Không lấy được dữ liệu từ SSI!")

    df = pd.concat(all_data, ignore_index=True)
    df = unify_prices(df)

    print(f"\n✅ Hoàn tất: {len(df)} records từ {df['ticker'].nunique()} mã")
    return df


def smart_fetch(tickers: List[str], start: str, end: str,
                source: str = 'auto') -> pd.DataFrame:
    """
    Smart fetch: tự động chọn nguồn tốt nhất
    source: 'auto', 'cafef', 'vnd', 'ssi'
    """
    if source == 'cafef':
        return fetch_prices_cafef(tickers, start, end)
    elif source == 'vnd':
        return fetch_prices_vnd(tickers, start, end)
    elif source == 'ssi':
        return fetch_prices_ssi(tickers, start, end)
    else:
        # Auto: CafeF → SSI → VNDirect
        print("🤖 Smart fetch: Ưu tiên CafeF...")
        try:
            return fetch_prices_cafef(tickers, start, end)
        except Exception as e1:
            print(f"⚠ CafeF fail: {str(e1)[:50]}")
            print("   Thử SSI...")
            try:
                return fetch_prices_ssi(tickers, start, end)
            except Exception as e2:
                print(f"⚠ SSI fail: {str(e2)[:50]}")
                print("   Thử VNDirect (cuối cùng)...")
                return fetch_prices_vnd(tickers, start, end)


def fetch_vnindex(start: str, end: str, source: str = 'auto') -> pd.DataFrame:
    """
    Lấy VNINDEX làm benchmark
    """
    print(f"📊 Đang tải VNINDEX...")

    if source == 'cafef' or source == 'auto':
        try:
            return fetch_prices_cafef(['VNINDEX'], start, end)
        except:
            if source != 'auto':
                raise

    if source == 'ssi' or source == 'auto':
        try:
            return fetch_prices_ssi(['VNINDEX'], start, end)
        except:
            if source != 'auto':
                raise

    # VNDirect fallback
    return fetch_prices_vnd(['VNINDEX'], start, end)


def load_from_csv(path: str, schema_map: Optional[Dict] = None) -> pd.DataFrame:
    """
    Đọc CSV người dùng
    Format: Ticker,Date,Open,High,Low,Close,Volume,OI
    """
    print(f"📂 Đang đọc CSV từ: {path}")

    try:
        df = pd.read_csv(path)

        default_map = {
            'Ticker': 'ticker',
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }

        if schema_map:
            default_map.update(schema_map)

        df = df.rename(columns=default_map)

        required = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"CSV thiếu các cột: {missing}")

        df = df[required]

        # Convert date YYYYMMDD -> datetime
        if df['date'].dtype == 'int64' or df['date'].dtype == 'object':
            df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d', errors='coerce')

        df = unify_prices(df)

        print(f"✅ Đọc thành công: {len(df)} records từ {df['ticker'].nunique()} mã")
        return df

    except FileNotFoundError:
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    except Exception as e:
        raise Exception(f"Lỗi đọc CSV: {str(e)}")


def unify_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa DataFrame"""
    df = df.copy()

    if df['date'].dtype != 'datetime64[ns]':
        df['date'] = pd.to_datetime(df['date'])

    df['ticker'] = df['ticker'].str.upper()
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    df = df.drop_duplicates(subset=['ticker', 'date'], keep='last')

    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['close'])

    for col in ['open', 'high', 'low']:
        df[col] = df[col].fillna(df['close'])

    df['volume'] = df['volume'].fillna(0)
    df = df.reset_index(drop=True)

    return df


def load_tickers_from_file(path: str) -> List[str]:
    """Đọc danh sách tickers từ file txt"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
        return tickers
    except FileNotFoundError:
        print(f"⚠ Không tìm thấy file: {path}")
        return []

import pandas as pd

def load_rf_investing_csv(path: str) -> pd.DataFrame:
    """
    Đọc CSV lợi suất TPCP 1 năm từ Investing
    và chuyển thành chuỗi lãi suất phi rủi ro theo tháng.

    Kỳ vọng CSV có:
        - Date: ngày (dd/mm/yyyy hoặc tương tự)
        - Price: lợi suất %/năm (ví dụ: 3.15)
    """
    df = pd.read_csv(path)

    # 1) Chuẩn hóa tên cột (sửa lại nếu file của bạn khác)
    df = df.rename(columns={
        "Date": "date",
        "Price": "yield_pct"
    })

    # 2) Chuyển kiểu ngày (Investing thường dùng dd/mm/yyyy)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)

    # 3) Từ % -> số thập phân (3.15% -> 0.0315)
    df["yield_annual"] = df["yield_pct"] / 100.0

    # 4) Resample theo tháng: lấy cuối tháng
    df_m = (
        df.set_index("date")
          .resample("M")
          .last()
          .reset_index()
    )

    # 5) Quy đổi từ annual -> monthly
    # Có thể dùng (1+r)^(1/12)-1, ở đây dùng đúng công thức:
    df_m["rf_monthly"] = (1.0 + df_m["yield_annual"])**(1/12.0) - 1.0

    # Trả về: date (cuối tháng), rf_monthly
    return df_m[["date", "rf_monthly"]]
