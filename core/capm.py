"""
Module phân tích CAPM với Newey-West SE
Fixed: Better error handling & debug logging
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from typing import Optional, Tuple


# ---------------------------------------------------
# Helper functions
# ---------------------------------------------------
def _std_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa tên cột"""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _to_datetime(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Chuyển cột date về datetime"""
    s = df[date_col]
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        s = s.astype("Int64").astype(str)
        dt = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    else:
        dt = pd.to_datetime(s, errors="coerce")
    df[date_col] = dt.dt.tz_localize(None) if hasattr(dt.dt, 'tz_localize') else dt
    return df


def _daily_log_return_from_close(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính log return từ close price
    """
    need = {"ticker", "date", "close"}
    if not need.issubset(df.columns):
        missing = need - set(df.columns)
        raise ValueError(f"Thiếu cột bắt buộc: {missing}")

    g = df.sort_values(["ticker", "date"]).copy()
    g["close"] = pd.to_numeric(g["close"], errors="coerce")
    g = g.dropna(subset=["close"])
    g = g[g["close"] > 0]

    # Tính log return
    g["ret"] = g.groupby("ticker")["close"].transform(
        lambda s: np.log(s / s.shift(1))
    )

    return g.dropna(subset=["ret"])


# ---------------------------------------------------
# 1) Resample lợi suất ngày → tháng
# ---------------------------------------------------
def resample_monthly(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Resample daily returns → monthly returns
    R_mth = sum(log returns trong tháng)
    """
    x = _std_cols(df)
    x = _to_datetime(x, "date")

    # Tính return nếu chưa có
    if "ret" not in x.columns:
        if verbose:
            print("   ℹ Tính log returns từ close prices...")
        x = _daily_log_return_from_close(x)

    x["month"] = x["date"].dt.to_period("M")

    out = (
        x.groupby(["ticker", "month"])["ret"]
        .sum()
        .reset_index()
        .rename(columns={"ret": "ret_m"})
    )

    out["date"] = out["month"].dt.to_timestamp("M")  # cuối tháng
    out = out[["ticker", "date", "ret_m"]].sort_values(["ticker", "date"])

    if verbose:
        print(f"   ✓ Resample monthly: {len(out)} month-ticker pairs")

    return out


# ---------------------------------------------------
# 2) Hồi quy CAPM cho 1 series
# ---------------------------------------------------
def run_capm_regression(
    returns: pd.Series,
    market_returns: pd.Series,
    rf_rate: float = 0.0,
    newey_west: bool = False
) -> dict:
    """
    Hồi quy: (R_i - r_f) = α + β (R_m - r_f) + ε
    """
    ri = pd.Series(returns).dropna().astype(float)
    rm = pd.Series(market_returns).dropna().astype(float)

    # Merge on common index
    df = pd.concat([ri.rename("ri"), rm.rename("rm")], axis=1).dropna()

    if df.empty:
        raise ValueError("run_capm_regression: không có dữ liệu giao nhau.")

    y = df["ri"] - rf_rate
    x = df["rm"] - rf_rate
    X = sm.add_constant(x.values)  # [const, (Rm - rf)]

    model = OLS(y.values, X)

    if newey_west:
        res = model.fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    else:
        res = model.fit()

    alpha, beta = res.params[0], res.params[1]
    se_a, se_b = res.bse[0], res.bse[1]
    t_a, t_b = res.tvalues[0], res.tvalues[1]
    p_a, p_b = res.pvalues[0], res.pvalues[1]
    r2 = res.rsquared

    # CI 95%
    try:
        ci = res.conf_int(alpha=0.05)
        a_lo, a_hi = ci[0, 0], ci[0, 1]
        b_lo, b_hi = ci[1, 0], ci[1, 1]
    except Exception:
        a_lo = a_hi = b_lo = b_hi = np.nan

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "alpha_se": float(se_a),
        "beta_se": float(se_b),
        "t_alpha": float(t_a),
        "t_beta": float(t_b),
        "p_alpha": float(p_a),
        "p_beta": float(p_b),
        "alpha_ci95": (float(a_lo), float(a_hi)),
        "beta_ci95": (float(b_lo), float(b_hi)),
        "R2": float(r2),
        "n": int(len(df))
    }


# ---------------------------------------------------
# 3) Phân tích CAPM cho nhiều mã
# ---------------------------------------------------
def capm_analysis(
    df: pd.DataFrame,
    market_df: pd.DataFrame,
    rf_mode: str = 'zero',
    newey_west: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    CAPM analysis cho nhiều stocks vs VNINDEX

    Args:
        df: Daily stock data (ticker, date, close hoặc ret)
        market_df: Daily VNINDEX data
        rf_mode: 'zero' | 'const' | 'series'
        newey_west: Dùng HAC standard errors
        verbose: Print debug info

    Returns:
        DataFrame với alpha, beta, R2, etc. cho mỗi ticker
    """
    if verbose:
        print("\n" + "="*60)
        print("🧮 CAPM ANALYSIS")
        print("="*60)

    # Chuẩn hóa & resample monthly
    if verbose:
        print("\n📊 Step 1: Resample to monthly...")

    stocks_m = resample_monthly(df, verbose=verbose)
    mk = _std_cols(market_df)

    # Ensure market has ticker column
    if "ticker" not in mk.columns:
        mk["ticker"] = "VNINDEX"

    market_m = resample_monthly(mk, verbose=verbose)

    # Get market series (prefer VNINDEX)
    mk_ticker = "VNINDEX"
    mk_series = (
        market_m[market_m["ticker"].str.upper() == mk_ticker]
        .set_index("date")["ret_m"]
        .sort_index()
    )

    if mk_series.empty:
        # Fallback: use first ticker
        if verbose:
            print(f"   ⚠ Không tìm thấy {mk_ticker}, dùng ticker đầu tiên")
        mk_series = market_m.groupby("date")["ret_m"].mean().sort_index()

    if verbose:
        print(f"\n   Market series: {len(mk_series)} months")
        print(f"   Date range: {mk_series.index.min()} → {mk_series.index.max()}")

    # r_f theo mode
    rf_const = 0.0
    rf_series = None

    if rf_mode == "const":
        if "rf" in mk.columns:
            try:
                mk_rf = mk.dropna(subset=["rf"]).copy()
                mk_rf["month"] = mk_rf["date"].dt.to_period("M")
                rf_const = float(mk_rf.groupby("month")["rf"].mean().mean())
                if verbose:
                    print(f"\n   Using const rf = {rf_const:.6f}")
            except Exception:
                rf_const = 0.0
        else:
            rf_const = 0.0

    elif rf_mode == "series":
        if "rf" in mk.columns:
            try:
                mk_rf = mk.dropna(subset=["rf"]).copy()
                # Lấy RF monthly (mean theo tháng), index là cuối tháng
                mk_rf["month"] = mk_rf["date"].dt.to_period("M")
                rf_series = (
                    mk_rf.groupby("month")["rf"]
                    .mean()
                )
                rf_series.index = rf_series.index.to_timestamp("M")
                if verbose:
                    print(f"\n   Using time-varying rf (series) with {len(rf_series)} months")
            except Exception:
                rf_series = None
        else:
            rf_series = None

    # Run CAPM for each stock
    if verbose:
        print(f"\n📈 Step 2: Running CAPM regressions...")
        print(f"   Total stock-months: {len(stocks_m)}")
        print(f"   Unique tickers: {stocks_m['ticker'].nunique()}")

    out_rows = []
    failed_tickers = []

    for tkr, g in stocks_m.groupby("ticker"):
        # Skip if ticker is market itself
        if tkr.upper() == mk_ticker:
            if verbose:
                print(f"   ⏭ {tkr}: Skipped (is market index)")
            continue

        s = g.set_index("date")["ret_m"].sort_index()

        # Align với market trên các tháng chung
        join = pd.concat([s.rename("ri"), mk_series.rename("rm")], axis=1).dropna()

        # Nếu dùng rf theo chuỗi thời gian -> merge thêm rf_series
        if rf_mode == "series" and rf_series is not None:
            join = join.join(rf_series.rename("rf"), how="inner")

        if len(join) < 6:
            failed_tickers.append((tkr, f"Insufficient data: {len(join)} months"))
            if verbose:
                print(f"   ✗ {tkr}: Chỉ có {len(join)} tháng chung (< 6)")
            continue

        try:
            if rf_mode == "series" and "rf" in join.columns:
                ri_excess = join["ri"] - join["rf"]
                rm_excess = join["rm"] - join["rf"]
                res = run_capm_regression(
                    returns=ri_excess,
                    market_returns=rm_excess,
                    rf_rate=0.0,
                    newey_west=newey_west
                )
            else:
                res = run_capm_regression(
                    returns=join["ri"],
                    market_returns=join["rm"],
                    rf_rate=(0.0 if rf_mode == "zero" else rf_const),
                    newey_west=newey_west
                )

            res.update({"ticker": tkr})
            out_rows.append(res)

            if verbose:
                print(f"   ✓ {tkr}: β={res['beta']:.3f}, R²={res['R2']:.3f}, n={res['n']}")


        except Exception as e:
            failed_tickers.append((tkr, str(e)))
            if verbose:
                print(f"   ✗ {tkr}: {str(e)[:50]}")

    # Summary
    if verbose:
        print("\n" + "="*60)
        print("📊 SUMMARY")
        print("="*60)
        print(f"   ✅ Success: {len(out_rows)} tickers")
        print(f"   ❌ Failed:  {len(failed_tickers)} tickers")

        if failed_tickers:
            print("\n   Failed tickers:")
            for tkr, reason in failed_tickers[:5]:  # Show first 5
                print(f"     • {tkr}: {reason}")
            if len(failed_tickers) > 5:
                print(f"     ... and {len(failed_tickers) - 5} more")

    # Return DataFrame
    if not out_rows:
        raise ValueError(
            f"CAPM không ước lượng được mã nào!\n"
            f"Lý do: {failed_tickers[0][1] if failed_tickers else 'Unknown'}\n"
            f"Gợi ý: Kiểm tra date range overlap giữa stocks và VNINDEX."
        )

    df_out = pd.DataFrame(out_rows).set_index("ticker").sort_index()

    if verbose:
        print(f"\n   ✅ CAPM analysis hoàn tất!")

    return df_out