"""
Module EDA - Exploratory Data Analysis
Fixed: tương thích với returns_clean.parquet từ clean.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

ASSETS_DIR = Path("assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def _std_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa tên cột"""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _pick_ret_col(df: pd.DataFrame) -> str:
    """
    Tìm cột lợi suất - ưu tiên theo thứ tự
    """
    candidates = [
        "return",  # từ clean.py
        "ret",  # từ clean.py
        "ret_w",  # nếu có winsorize riêng
        "log_ret",
        "log_return",
        "simple_ret",
        "returns",
        "r"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return ""


def _ensure_returns(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Đảm bảo có cột lợi suất để vẽ
    """
    df = _std_cols(df)
    col = _pick_ret_col(df)

    if col:
        return df, col

    # Tự tính từ close nếu cần
    need = {"ticker", "date", "close"}
    if need.issubset(df.columns):
        g = df.sort_values(["ticker", "date"]).copy()
        g["close"] = pd.to_numeric(g["close"], errors="coerce")
        g = g[g["close"] > 0]
        g["return"] = g.groupby("ticker")["close"].transform(
            lambda s: np.log(s / s.shift(1))
        )
        g = g.dropna(subset=["return"])
        return g, "return"

    raise KeyError(
        "Không tìm thấy cột lợi suất (return/ret) "
        "và cũng không có đủ cột để tự tính (cần: ticker, date, close)."
    )


def _clip_by_date(df: pd.DataFrame, start: Optional[pd.Timestamp],
                  end: Optional[pd.Timestamp]) -> pd.DataFrame:
    """Lọc theo khoảng thời gian"""
    if start is not None:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end is not None:
        df = df[df["date"] <= pd.to_datetime(end)]
    return df


def line_price(
        prices: pd.DataFrame,
        tickers: List[str],
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        log_scale: bool = False,
        fname_suffix: str = ""
) -> Path:
    """
    Vẽ biểu đồ giá theo thời gian
    """
    df = _std_cols(prices.copy())

    # Ensure date column
    if "date" not in df.columns:
        raise ValueError("Cần cột 'date'")

    df["date"] = pd.to_datetime(df["date"])
    df = df[df["ticker"].str.upper().isin([t.upper() for t in tickers])]
    df = _clip_by_date(df, start, end)

    if df.empty:
        raise ValueError("line_price: không có dữ liệu sau khi lọc.")

    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)

    for t in sorted(df["ticker"].unique()):
        g = df[df["ticker"] == t].sort_values("date")
        ax.plot(g["date"], g["close"], label=t, linewidth=2, marker='o',
                markersize=3, alpha=0.8)

    if log_scale:
        ax.set_yscale("log")

    ax.set_title(f"📈 Close Price — {', '.join(tickers)}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Close Price", fontsize=11)
    ax.legend(ncol=3, fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    fname = f"line_price_{'_'.join(sorted(tickers))}{fname_suffix}.png"
    out = ASSETS_DIR / fname
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"   💾 Saved: {out}")
    return out


def hist_returns(
        returns: pd.DataFrame,
        tickers: List[str],
        bins: int = 50,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        fname_suffix: str = ""
) -> Path:
    """
    Vẽ histogram phân phối lợi suất
    """
    df, col = _ensure_returns(returns.copy())

    df = df[df["ticker"].str.upper().isin([t.upper() for t in tickers])]
    df = _clip_by_date(df, start, end)

    if df.empty:
        raise ValueError("hist_returns: không có dữ liệu.")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    for t in sorted(df["ticker"].unique()):
        g = df[df["ticker"] == t]
        ret_data = g[col].dropna()
        ax.hist(ret_data, bins=bins, alpha=0.4, label=t, density=True, edgecolor='black')

    ax.set_title(f"📊 Histogram of Returns — {', '.join(tickers)}",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel(f"{col.upper()}", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(ncol=2, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    fname = f"hist_returns_{'_'.join(sorted(tickers))}{fname_suffix}.png"
    out = ASSETS_DIR / fname
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"   💾 Saved: {out}")
    return out


def boxplot_by_month(
        returns: pd.DataFrame,
        tickers: List[str],
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        fname_suffix: str = ""
) -> Path:
    """
    Vẽ boxplot lợi suất theo tháng
    """
    df, col = _ensure_returns(returns.copy())

    df = df[df["ticker"].str.upper().isin([t.upper() for t in tickers])]
    df = _clip_by_date(df, start, end)

    if df.empty:
        raise ValueError("boxplot_by_month: không có dữ liệu.")

    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(14, 6), dpi=100)

    sns.boxplot(data=df, x="month", y=col, hue="ticker", ax=ax, palette="Set2")

    ax.set_title(f"📦 Monthly Return Distribution — {', '.join(tickers)}",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel(f"{col.upper()}", fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=len(tickers), fontsize=9, loc='upper right')

    fname = f"boxplot_month_{'_'.join(sorted(tickers))}{fname_suffix}.png"
    out = ASSETS_DIR / fname
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"   💾 Saved: {out}")
    return out


def corr_heatmap(
        returns: pd.DataFrame,
        tickers: List[str],
        method: str = "pearson",
        min_periods: int = 30,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        fname_suffix: str = ""
) -> Tuple[Path, pd.DataFrame]:
    """
    Vẽ heatmap ma trận tương quan
    """
    df, col = _ensure_returns(returns.copy())

    df = df[df["ticker"].str.upper().isin([t.upper() for t in tickers])]
    df = _clip_by_date(df, start, end)

    if df.empty:
        raise ValueError("corr_heatmap: không có dữ liệu.")

    # Pivot: date x ticker
    piv = df.pivot_table(index="date", columns="ticker", values=col, aggfunc="first")

    # Drop rows with any missing (đảm bảo cùng mẫu)
    piv = piv.dropna(how="any")

    if piv.shape[0] < min_periods:
        print(f"   ⚠ Chỉ có {piv.shape[0]} ngày chung (< {min_periods}), correlation có thể không ổn định")

    corr = piv.corr(method=method, min_periods=min_periods)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=100)

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Correlation"},
        linewidths=0.5,
        square=True
    )

    ax.set_title(
        f"🔥 {method.title()} Correlation — {', '.join(tickers)}\n"
        f"(Based on {piv.shape[0]} common trading days)",
        fontsize=13, fontweight='bold'
    )

    fname = f"corr_{method}_{'_'.join(sorted(tickers))}{fname_suffix}.png"
    out = ASSETS_DIR / fname
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"   💾 Saved: {out}")
    return out, corr


def summary_stats(returns: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Tính thống kê mô tả cho lợi suất
    """
    df, col = _ensure_returns(returns.copy())
    df = df[df["ticker"].str.upper().isin([t.upper() for t in tickers])]

    stats = df.groupby("ticker")[col].agg([
        ("Count", "count"),
        ("Mean", "mean"),
        ("Std", "std"),
        ("Min", "min"),
        ("25%", lambda x: x.quantile(0.25)),
        ("Median", "median"),
        ("75%", lambda x: x.quantile(0.75)),
        ("Max", "max"),
        ("Skew", lambda x: x.skew()),
        ("Kurt", lambda x: x.kurtosis())
    ]).round(6)

    return stats