"""
Portfolio analytics: beta-sorted portfolios, Sharpe, equity curves, benchmark VNINDEX.
Giả định:
- Giá & VNINDEX lấy từ CafeF (parquet) → có ['ticker','date','close', ...]
- Kết quả CAPM đã tính → DataFrame index=ticker, có ['beta','R2', 'n', ...]
- Nếu chưa có lợi suất tháng, module sẽ tự resample từ giá (log-return).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# ---------------------------
# Helpers (tái sử dụng logic)
# ---------------------------
def _std_cols(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [c.strip().lower() for c in x.columns]
    return x

def _to_datetime(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    s = df[date_col]
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        s = s.astype("Int64").astype(str)
        dt = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    else:
        dt = pd.to_datetime(s, errors="coerce")
    df[date_col] = dt.dt.tz_localize(None)
    return df

def _daily_log_return_from_close(df: pd.DataFrame) -> pd.DataFrame:
    need = {"ticker", "date", "close"}
    if not need.issubset(df.columns):
        raise ValueError(f"Thiếu cột bắt buộc: {need - set(df.columns)}")
    g = df.sort_values(["ticker","date"]).copy()
    g["ticker"] = g["ticker"].astype(str)                   # ổn định groupby
    g["close"]  = pd.to_numeric(g["close"], errors="coerce")
    g = g.dropna(subset=["close"])
    g = g[g["close"] > 0]
    g["ret_d"] = g.groupby("ticker")["close"].transform(lambda s: np.log(s / s.shift(1)))
    return g.dropna(subset=["ret_d"])

def _monthly_from_daily(df: pd.DataFrame, col: str = "ret_d", out_col: str = "ret_m") -> pd.DataFrame:
    x = df.copy()
    x["month"] = x["date"].dt.to_period("M")
    out = (
        x.groupby(["ticker","month"])[col]
         .sum()
         .reset_index()
         .rename(columns={col: out_col})
    )
    out["date"] = out["month"].dt.to_timestamp("M")
    out = out[["ticker","date",out_col]].sort_values(["ticker","date"])
    return out

# ---------------------------
# 1) Chuẩn bị lợi suất tháng cổ phiếu & chỉ số
# ---------------------------
def monthly_returns_from_prices(stocks: pd.DataFrame, vnindex: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    stocks: giá ngày nhiều mã (CafeF) → ['ticker','date','close', ...]
    vnindex: giá ngày chỉ số → ['date','close'] (có thể có ticker hoặc không)
    Trả: (df_monthly, vni_monthly_series)
        - df_monthly: ['ticker','date','ret_m']
        - vni_series: pd.Series index=date, name='rm'
    """
    st = _std_cols(stocks)
    st = _to_datetime(st, "date")
    if "ret_m" not in st.columns:
        st = _daily_log_return_from_close(st)
        st = _monthly_from_daily(st, col="ret_d", out_col="ret_m")
    else:
        st = st[["ticker","date","ret_m"]].copy()
        st = _to_datetime(st, "date")

    mk = _std_cols(vnindex)
    mk = _to_datetime(mk, "date")
    if "ret_m" not in mk.columns:
        mk = _daily_log_return_from_close(mk)
        mk = _monthly_from_daily(mk, col="ret_d", out_col="ret_m")
    else:
        mk = mk[["date","ret_m"]].copy()
    # chọn series market
    if "ticker" in mk.columns:
        mk_series = mk.sort_values("date").set_index("date")
        # nếu nhiều ticker, chọn VNINDEX; nếu không, mean theo ngày
        if "vnindex" in mk["ticker"].str.lower().unique():
            mk_series = mk_series[mk_series["ticker"].str.lower()=="vnindex"]["ret_m"]
        else:
            mk_series = mk.groupby("date")["ret_m"].mean()
    else:
        mk_series = mk.sort_values("date").set_index("date")["ret_m"]

    mk_series.name = "rm"
    return st, mk_series

# ---------------------------
# 2) Xây portfolio β-sorted
# ---------------------------
def assign_beta_bucket(capm_res: pd.DataFrame, n_quantiles: int = 4) -> pd.DataFrame:
    """
    capm_res: index=ticker, có cột 'beta'.
    Trả về DataFrame có cột 'beta_q' (1..n_quantiles).
    """
    df = capm_res.copy()
    if "beta" not in df.columns:
        raise ValueError("capm_res thiếu cột 'beta'.")
    df = df[["beta"]].dropna()
    # qcut có thể lỗi nếu beta trùng quá nhiều → dùng rank làm tie-break
    rk = df["beta"].rank(method="average")
    df["beta_q"] = pd.qcut(rk, q=n_quantiles, labels=range(1, n_quantiles+1))
    return df

def backtest_portfolios(
    monthly_returns: pd.DataFrame,
    beta_buckets: pd.DataFrame,
    market_series: pd.Series,
    weight_mode: str = "equal"
) -> dict:
    """
    monthly_returns: ['ticker','date','ret_m']
    beta_buckets: index=ticker, columns=['beta_q']
    market_series: Series (date-index) ret_m của VNINDEX
    weight_mode: 'equal' | 'liquidity' (nếu monthly có cột 'volume_m' hoặc 'turnover_m')
    Trả: dict {'Q1': equity_curve, 'Qn': equity_curve, 'MKT': equity_curve}
    """
    # merge bucket vào monthly
    df = monthly_returns.merge(beta_buckets.reset_index(), on="ticker", how="inner")
    df = df.dropna(subset=["ret_m","beta_q"]).copy()
    # tạo trọng số
    if weight_mode == "equal":
        df["w"] = 1.0
    else:
        # proxy thanh khoản nếu có, nếu không fallback equal
        for c in ["turnover_m","volume_m"]:
            if c in df.columns:
                v = df.groupby(["beta_q","date"])[c].transform(lambda x: x / x.sum())
                df["w"] = v.fillna(0.0)
                break
        if "w" not in df.columns:
            df["w"] = 1.0

    # chuẩn hoá theo nhóm (mỗi tháng trong mỗi bucket tổng w = 1)
    df["w"] = df.groupby(["beta_q","date"])["w"].transform(lambda x: x / x.sum())
    df["w"] = df["w"].fillna(0.0)

    # tính lợi suất danh mục theo bucket
    port = (
        df.groupby(["beta_q","date"])
          .apply(lambda g: np.sum(g["w"] * g["ret_m"]))
          .rename("ret_p")
          .reset_index()
    )
    # equity curves (log-returns → cộng dồn rồi exp)
    out = {}
    for q, g in port.groupby("beta_q"):
        s = g.set_index("date")["ret_p"].sort_index()
        eq = np.exp(s.cumsum())

        # q có thể là số (1,2,3,4) hoặc chuỗi ("Q1","Q_Low",...)
        if isinstance(q, (int, float, np.integer)):
            name = f"Q{int(q)}"
        else:
            # nếu đã là "Q1", "Q_High"… thì giữ nguyên
            name = str(q)

        eq.name = name
        out[name] = eq

    # market equity
    mkt = market_series.sort_index()
    mkt_eq = np.exp(mkt.cumsum())
    mkt_eq.name = "MKT"
    out["MKT"] = mkt_eq
    return out

# ---------------------------
# 3) Chỉ số hiệu quả
# ---------------------------
def metrics_from_returns(ret: pd.Series, rf: float = 0.0, periods_per_year: int = 12) -> dict:
    """
    ret: monthly log-returns
    """
    r = ret.dropna()
    if r.empty:
        return {"ann_ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "mdd": np.nan, "n": 0}
    ann_ret = r.mean() * periods_per_year
    ann_vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
    # max drawdown theo equity
    eq = np.exp(r.cumsum())
    peak = eq.cummax()
    dd = (eq / peak - 1.0).min()
    return {
        "ann_ret": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe":  float(sharpe),
        "mdd":     float(dd),
        "n":       int(len(r))
    }

def summarize_portfolios(monthly_returns: pd.DataFrame, curves: dict) -> pd.DataFrame:
    """
    Tạo bảng Sharpe/Return/Vol/MDD cho từng portfolio + MKT.
    """
    # dựng returns lại từ equity curves bằng diff log
    out_rows = []
    for name, eq in curves.items():
        eq = eq.dropna().sort_index()
        r = np.log(eq / eq.shift(1)).dropna()
        m = metrics_from_returns(r, rf=0.0, periods_per_year=12)
        m.update({"portfolio": name})
        out_rows.append(m)
    res = pd.DataFrame(out_rows).set_index("portfolio").sort_index()
    return res.round(4)
