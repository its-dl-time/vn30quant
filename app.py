"""
VN30 Quantitative Analysis Dashboard
Streamlit app với cấu trúc 2 Tabs: Phân tích & Chiến lược
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from datetime import datetime

# Import core modules
from core.data_io import fetch_prices_cafef, smart_fetch, fetch_vnindex
from core.clean import clean_pipeline, get_return_statistics
from core.eda import line_price, hist_returns, boxplot_by_month, corr_heatmap, summary_stats
from core.arima import (
    check_stationarity,
    fit_arima_on_returns,
    forecast_arima_returns,
    forecast_figure,
    rolling_backtest
)
from core.capm import capm_analysis
from core.portfolio import (
    monthly_returns_from_prices,
    assign_beta_bucket,
    backtest_portfolios,
    summarize_portfolios
)
from core.report import build_pdf_report
from core.data_manager import DataManager, smart_load_data
from core.data_io import load_rf_investing_csv
import os

os.makedirs("assets", exist_ok=True)   # Lưu ảnh biểu đồ

# Page config
st.set_page_config(
    page_title="VN30 Quant Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)

ASSETS_DIR = Path("assets")
ASSETS_DIR.mkdir(exist_ok=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'prices_df' not in st.session_state:
    st.session_state.prices_df = None
if 'returns_df' not in st.session_state:
    st.session_state.returns_df = None
if 'vnindex_df' not in st.session_state:
    st.session_state.vnindex_df = None
if 'capm_results' not in st.session_state:
    st.session_state.capm_results = None
if 'special_ticker' not in st.session_state:
    st.session_state.special_ticker = ""
if 'rf_df' not in st.session_state:
    st.session_state.rf_df = None

# ============================================================================
# CSS STYLING
# ============================================================================
st.markdown("""
    <style>
    /* Sidebar styling giữ nguyên */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f36 0%, #0f1419 100%);
    }

    /* NEW: Style cho Header to và sáng */
    h1 {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        background: -webkit-linear-gradient(45deg, #00C853, #64B5F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 10px;
    }
    h2 {
        font-size: 1.8rem !important;
        color: #4CAF50 !important;
        border-bottom: 1px solid #333;
        padding-top: 20px;
        padding-bottom: 10px;
    }
    h3 {
        font-size: 1.4rem !important;
        color: #64B5F6 !important;
        font-weight: 600 !important;
    }

    /* Style cho Tab Header chữ to */
    .stTabs [data-baseweb="tab"] {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        color: #B0BEC5;
    }
    .stTabs [aria-selected="true"] {
        color: #4CAF50 !important;
        border-bottom-color: #4CAF50 !important;
    }

    /* Table styling */
    [data-testid="stDataFrame"] {
        background-color: #1a1c24;
        border-radius: 10px;
        padding: 5px;
    }

    /* Sidebar elements styles (Giữ lại style cũ của bạn) */
    .sidebar-section { color: #8b92a7; font-size: 0.75rem; text-transform: uppercase; font-weight: 600; margin-top: 1.5rem; }
    .status-box { background: rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 0.75rem; border-left: 3px solid #4CAF50; margin-bottom: 0.5rem; }
    .status-value { color: #ffffff; font-size: 0.9rem; font-weight: 500; }
    .cache-fresh { color: #4CAF50; } .cache-stale { color: #ff9800; }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# SIDEBAR LOGIC
# ============================================================================

def load_vn30_tickers() -> list[str]:
    """Load VN30 tickers từ file"""
    try:
        with open('config/tickers_vn30.txt', 'r') as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
        return tickers
    except:
        return ['ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR',
                'HDB', 'HPG', 'KDH', 'MBB', 'MSN', 'MWG', 'NVL', 'PDR',
                'PLX', 'POW', 'SAB', 'SHB', 'SSI', 'STB', 'TCB', 'TPB',
                'VCB', 'VHM', 'VIC', 'VJC', 'VNM', 'VPB']


def sidebar_data_loading():

    """Modern sidebar: Tự động tải dữ liệu & Cập nhật"""
    st.sidebar.markdown("### 📊 VN30 Quant Analysis")
    st.sidebar.markdown("---")

    # 1. Cấu hình Dữ liệu
    st.sidebar.markdown('<div class="sidebar-section">📋 Cấu hình dữ liệu</div>', unsafe_allow_html=True)

    vn30_tickers = load_vn30_tickers()
    with st.sidebar.expander("ℹ️ Rổ VN30", expanded=False):
        st.markdown(f"**{len(vn30_tickers)} mã cổ phiếu:**")
        cols_per_row = 5
        for i in range(0, len(vn30_tickers), cols_per_row):
            row_tickers = vn30_tickers[i:i + cols_per_row]
            st.text(" • ".join(row_tickers))

    # Input Mã đặc biệt
    special_ticker = st.sidebar.text_input(
        "🎯 Mã đặc biệt (ARIMA):", value="GAS", max_chars=10
    ).strip().upper()

    # Cập nhật session state ngay khi nhập
    if special_ticker != st.session_state.get('special_ticker', ''):
        st.session_state.special_ticker = special_ticker

    # Input Thời gian
    st.sidebar.markdown('<div class="sidebar-section">📅 Khoảng thời gian</div>', unsafe_allow_html=True)
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Từ ngày:", value=pd.to_datetime("2020-01-01"))
    with col2:
        end_date = st.date_input("Đến ngày:", value=datetime.now())

    st.sidebar.markdown("---")

    # Chuẩn bị danh sách ticker
    all_tickers = list(set(vn30_tickers + [special_ticker]))

    # Hàm nội bộ để thực hiện việc tải dữ liệu (tránh lặp code)
    def execute_load_data(is_refresh=False):
        loading_text = "🔄 Đang làm mới dữ liệu..." if is_refresh else "⏳ Đang tự động tải dữ liệu..."
        with st.spinner(loading_text):
            try:
                # Nếu là refresh thì xóa cache cũ để tải mới
                if is_refresh:
                    for f in CACHE_DIR.glob("*.parquet"):
                        try:
                            f.unlink()
                        except:
                            pass

                # Gọi hàm Smart Load
                data = smart_load_data(
                    tickers=all_tickers,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    load_vnindex=True,
                    cache_dir=str(CACHE_DIR),
                    max_cache_age_days=1
                )

                # Lưu vào Session State
                st.session_state.prices_df = data['prices_clean']
                st.session_state.returns_df = data['returns']
                st.session_state.vnindex_df = data.get('vnindex')
                st.session_state.special_ticker = special_ticker

                # Load RF từ CSV
                try:
                    rf_path = "data_cache/bond_vn_1y_investing.csv"
                    if Path(rf_path).exists():
                        rf_raw = load_rf_investing_csv(rf_path)
                        mask = (
                                (rf_raw["date"] >= pd.to_datetime(start_date)) &
                                (rf_raw["date"] <= pd.to_datetime(end_date))
                        )
                        st.session_state.rf_df = rf_raw.loc[mask].reset_index(drop=True)
                    else:
                        st.session_state.rf_df = None

                except Exception as e:
                    st.sidebar.warning(f"RF Warning: {e}")
                    st.session_state.rf_df = None

                # --- [AUTO SAVE - ĐỒNG BỘ PARQUET] ---
                # Lưu vào CACHE_DIR để các Tab khác và PDF dùng chung
                try:
                    # Đảm bảo thư mục tồn tại
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)

                    # Lưu Prices
                    data['prices_clean'].to_parquet(CACHE_DIR / "prices_clean.parquet")
                    data['prices_clean'].to_csv("assets/prices_clean.csv")


                    # Lưu Returns
                    data['returns'].to_parquet(CACHE_DIR / "returns.parquet")
                    data['prices_clean'].to_csv("assets/returns.csv")


                    # Lưu VNINDEX
                    if data.get('vnindex') is not None:
                        data.get('vnindex').to_parquet(CACHE_DIR / "vnindex.parquet")
                        data.get('vnindex').to_csv("assets/vnindex.csv")

                    # Lưu Risk Free (nếu có)
                    if st.session_state.rf_df is not None:
                        st.session_state.rf_df.to_parquet(CACHE_DIR / "rf_data.parquet")

                except Exception as save_e:
                    print(f"Lỗi lưu Cache Parquet: {save_e}")
                # ----------------------------------------------
                st.sidebar.success("✅ Đã tải dữ liệu!")

                # Rerun phải là lệnh cuối cùng
                if is_refresh:
                    st.rerun()

            except Exception as e:
                st.sidebar.error(f"❌ Lỗi tải dữ liệu: {str(e)}")

    # 2. LOGIC TỰ ĐỘNG TẢI (AUTO LOAD)
    # Nếu chưa có dữ liệu trong Session -> Tự động chạy
    if st.session_state.prices_df is None:
        execute_load_data(is_refresh=False)

    # 3. NÚT TẢI LẠI (REFRESH)
    st.sidebar.markdown('<div class="sidebar-section">🚀 Tác vụ</div>', unsafe_allow_html=True)
    if st.sidebar.button("🔄 Tải lại / Cập nhật Dữ liệu", type="primary", use_container_width=True):
        execute_load_data(is_refresh=True)

    # 4. HIỂN THỊ TRẠNG THÁI (STATUS)
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sidebar-section">📊 Trạng thái Dữ liệu</div>', unsafe_allow_html=True)

    if st.session_state.prices_df is not None:
        row_count = len(st.session_state.prices_df)
        st.sidebar.markdown(
            f'<div class="status-box"><div class="status-label">💰 PRICES LOADED</div>'
            f'<div class="status-value">{row_count:,} rows</div></div>',
            unsafe_allow_html=True
        )
        # Hiển thị thêm thông tin cache nếu có
        manager = DataManager(CACHE_DIR)
        meta = manager.get_cache_info().get('metadata', {})
        if 'prices_last_date' in meta:
            st.sidebar.caption(f"🕒 Cache time: {meta['prices_last_date']}")
    else:
        st.sidebar.markdown('<div class="status-box pending"><div class="status-value">No Data</div></div>',
                            unsafe_allow_html=True)

# ============================================================================
# FUNCTION BLOCKS (LOGIC)
# ============================================================================

def tab_eda():
    """EDA Section: Layout 3 cột - Đồng bộ mã với Sidebar"""
    st.header("1. Phân tích Dữ liệu Khám phá (EDA)")

    # 1. Kiểm tra dữ liệu
    if st.session_state.prices_df is None:
        st.warning("⚠️ Vui lòng tải dữ liệu từ sidebar trước!")
        return

    prices = st.session_state.prices_df
    import plotly.express as px

    # --- [BƯỚC 1] LÀM SẠCH DỮ LIỆU ---
    # Loại bỏ các dòng giá <= 0 (Lỗi dữ liệu)
    prices_clean = prices[prices['close'] > 0].copy()

    # Pivot dữ liệu
    pivot_prices = prices_clean.pivot_table(index='date', columns='ticker', values='close')

    # Kiểm tra dòng cuối cùng: Nếu giá trị bằng 0 hoặc NaN thì cắt bỏ
    if not pivot_prices.empty:
        last_row = pivot_prices.iloc[-1]
        if (last_row == 0).any() or last_row.isna().all():
            pivot_prices = pivot_prices.iloc[:-1]

    # Tính Return và xử lý vô cực
    daily_returns = pivot_prices.pct_change()
    daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()

    all_tickers = sorted(prices_clean['ticker'].unique())

    # --- [ĐÃ XÓA ĐOẠN KHAI BÁO LẠI DỮ LIỆU THÔ Ở ĐÂY ĐỂ TRÁNH LỖI] ---

    # --- [BƯỚC 2] CẤU HÌNH & VISUALIZATION ---
    # Lấy mã đang nhập ở Sidebar
    sidebar_ticker = st.session_state.get('special_ticker', 'GAS')

    # Kiểm tra nếu mã sidebar có trong dữ liệu thì lấy
    default_selections = [sidebar_ticker] if sidebar_ticker in all_tickers else [all_tickers[0]]

    selected_tickers = st.multiselect(
        "Chọn mã phân tích:",
        all_tickers,
        default=default_selections
    )

    if selected_tickers:
        # Lọc dữ liệu vẽ biểu đồ (DÙNG PRICES_CLEAN thay vì PRICES thô)
        subset_price = prices_clean[prices_clean['ticker'].isin(selected_tickers)]

        # Unpivot để vẽ nhiều đường
        subset_ret = daily_returns[selected_tickers].reset_index().melt(id_vars='date', var_name='ticker',
                                                                        value_name='return')

        # --- VISUALIZATION (3 CỘT GỌN GÀNG) ---
        st.markdown("#### 📊 Biểu đồ Trực quan")
        c1, c2, c3 = st.columns(3)


        # Cột 1: Giá (Price)
        with c1:
            st.caption("1. Diễn biến Giá")
            fig_p = px.line(subset_price, x='date', y='close', color='ticker', template="plotly_dark", height=300)
            fig_p.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0), xaxis_title=None, yaxis_title=None)
            fig_p.write_image("assets/eda_price.png")  # <--- LƯU
            st.plotly_chart(fig_p, use_container_width=True)

        # Cột 2: Biến động (Return Volatility)
        with c2:
            st.caption("2. Biến động Lợi suất")
            fig_r = px.line(subset_ret, x='date', y='return', color='ticker', template="plotly_dark", height=300)
            fig_r.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0), xaxis_title=None, yaxis_title=None)
            fig_r.write_image("assets/eda_return.png")  # <--- LƯU
            st.plotly_chart(fig_r, use_container_width=True)

        # Cột 3: Phân phối (Histogram)
        with c3:
            st.caption("3. Phân phối Tần suất")
            fig_h = px.histogram(subset_ret, x="return", color="ticker", barmode="overlay", opacity=0.6,
                                 template="plotly_dark", height=300)
            fig_h.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0), xaxis_title=None, yaxis_title=None)
            fig_h.write_image("assets/eda_hist.png")  # <--- LƯU
            st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("---")

    # --- THỐNG KÊ & HEATMAP ---
    st.markdown("#### 📋 Thống kê Tổng hợp & Tương quan")

    # Bảng thống kê (Full 30 mã)
    stats_df = daily_returns.describe().T[['mean', 'std', 'min', 'max', '50%']]
    stats_df.columns = ['Mean', 'Std', 'Min', 'Max', 'Median']
    stats_df['Skew'] = daily_returns.skew()
    stats_df.to_csv("assets/eda_summary_stats.csv")  # <--- LƯU CSV

    # Layout 2 cột
    col_tbl, col_hm = st.columns([1.5, 1])

    with col_tbl:
        st.dataframe(
            stats_df.sort_values('Mean', ascending=False).style.format("{:.4f}").background_gradient(cmap='Greens',
                                                                                                     subset=['Mean']),
            use_container_width=True, height=400
        )

    with col_hm:
        corr = daily_returns.corr()
        fig_corr = px.imshow(corr, text_auto=False, color_continuous_scale='RdBu_r')
        fig_corr.write_image("assets/eda_heatmap.png")  # <--- LƯU ẢNH
        fig_corr.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_corr, use_container_width=True)


def tab_capm():
    """CAPM Section: Auto-run logic + Manual refresh + Full Charts"""
    # 1. Kiểm tra dữ liệu đầu vào
    if st.session_state.prices_df is None or st.session_state.vnindex_df is None:
        st.warning("⚠️ Cần dữ liệu VNINDEX. Vui lòng tải lại dữ liệu.")
        return

    stocks = st.session_state.prices_df
    vnindex = st.session_state.vnindex_df

    # --- CẤU HÌNH ---
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        rf_mode = st.selectbox("Risk-free Rate:", ["series", "zero"], index=0)
    with col2:
        newey_west = st.checkbox("Newey-West SE", value=True)
    with col3:
        st.write("")
        manual_run = st.button("🔄 Chạy lại CAPM", key="btn_capm", use_container_width=True)

    # --- LOGIC TỰ ĐỘNG CHẠY ---
    should_run = manual_run or (st.session_state.capm_results is None)

    if should_run:
        with st.spinner("Đang tự động tính toán Beta cho 30 mã..."):
            try:
                res = capm_analysis(stocks, vnindex, rf_mode=rf_mode, newey_west=newey_west)
                if res is not None and not res.empty:
                    st.session_state.capm_results = res
                    # Lưu cache và CSV
                    res.to_parquet(CACHE_DIR / "capm_results_app.parquet")
                    res.to_csv("exports/capm_results.csv")

                    if manual_run:
                        st.success(f"✅ Đã cập nhật: {len(res)} mã.")
                else:
                    st.error("Lỗi: Không tính được kết quả.")
            except Exception as e:
                st.error(f"Lỗi tính toán: {e}")

    # --- HIỂN THỊ KẾT QUẢ ---
    if st.session_state.capm_results is not None:
        capm_res = st.session_state.capm_results
        import plotly.express as px

        # Bảng kết quả
        st.dataframe(
            capm_res[['alpha', 'beta', 'R2', 'p_beta']].style.format("{:.4f}").background_gradient(subset=['beta'],
                                                                                                   cmap="Blues"),
            use_container_width=True, height=250
        )

        st.markdown("---")

        # Chart 1: Beta vs R2
        st.markdown("### 1. Beta vs. R² (Độ tin cậy)")
        fig1 = px.scatter(
            capm_res, x="beta", y="R2",
            text=capm_res.index if 'ticker' not in capm_res.columns else capm_res['ticker'],
            color="R2", color_continuous_scale="Viridis",
            height=450
        )
        fig1.update_traces(textposition='top center', marker=dict(size=12))
        fig1.update_layout(template="plotly_dark")
        fig1.write_image("assets/capm_beta_r2.png")  # Lưu ảnh
        st.plotly_chart(fig1, use_container_width=True)

        with st.expander("💡 Giải thích"):
            st.caption("R² càng cao (gần 1) thì Beta càng đáng tin cậy.")

        st.markdown("---")

        # Chart 2: Alpha vs Beta
        st.markdown("### 2. Alpha vs. Beta (Hiệu suất thực)")
        fig2 = px.scatter(
            capm_res, x="beta", y="alpha",
            text=capm_res.index if 'ticker' not in capm_res.columns else capm_res['ticker'],
            color="alpha", color_continuous_scale="RdYlGn",
            height=450
        )
        fig2.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        fig2.add_vline(x=1, line_dash="dot", line_color="gray", opacity=0.5)
        fig2.update_traces(textposition='top center', marker=dict(size=12))
        fig2.update_layout(template="plotly_dark")
        fig2.write_image("assets/capm_alpha_beta.png")  # Lưu ảnh
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("💡 Giải thích"):
            st.caption("Góc Trái-Trên: Lợi nhuận cao (Alpha > 0), Rủi ro thấp (Beta < 1).")

        st.markdown("---")

        # --- [MỚI] CHART 3: BETA RANKING (BAR CHART) ---
        st.markdown("### 3. Xếp hạng Beta (Mức độ Rủi ro)")

        # Chuẩn bị dữ liệu vẽ (Sắp xếp giảm dần)
        df_plot = capm_res.sort_values('beta', ascending=False)
        # Đảm bảo có cột ticker để vẽ trục X
        if 'ticker' not in df_plot.columns:
            df_plot = df_plot.reset_index()
            # Nếu reset index mà tên cột index cũ là 'index' hoặc None thì đổi thành 'ticker'
            if 'ticker' not in df_plot.columns:
                df_plot.columns.values[0] = 'ticker'

        fig3 = px.bar(
            df_plot,
            x='ticker',
            y='beta',
            color='beta',
            color_continuous_scale='Spectral_r',  # Màu đỏ (cao) -> Xanh (thấp)
            text_auto='.2f',
            height=500
        )

        # Thêm đường tham chiếu Beta = 1
        fig3.add_hline(y=1, line_dash="dash", line_color="white", annotation_text="Market Risk (1.0)")

        fig3.update_layout(
            template="plotly_dark",
            xaxis_title=None,
            yaxis_title="Beta Hệ thống",
            hovermode="x unified"
        )

        # Lưu ảnh và hiển thị
        fig3.write_image("assets/capm_beta_bar.png")
        st.plotly_chart(fig3, use_container_width=True)

def tab_arima():
    """ARIMA Section: Auto-Run, AIC/BIC Selection & Display"""

    # 1. Kiểm tra dữ liệu
    if st.session_state.prices_df is None:
        st.warning("⚠️ Đang chờ dữ liệu...")
        return

    prices_df = st.session_state.prices_df
    all_tickers = sorted(prices_df['ticker'].unique())

    # Layout 2 cột
    col_settings, col_results = st.columns([0.3, 0.7])

    with col_settings:
        st.markdown("#### ⚙️ Thiết lập")
        default_ticker = st.session_state.get('special_ticker', 'GAS')
        if default_ticker not in all_tickers: default_ticker = all_tickers[0]

        ticker = st.selectbox("Mã CK", all_tickers, index=all_tickers.index(default_ticker))

        # --- XỬ LÝ DỮ LIỆU ---
        ticker_data = prices_df[prices_df['ticker'] == ticker].sort_values('date')
        ticker_data = ticker_data[ticker_data['close'] > 0].dropna(subset=['close'])
        price_series = ticker_data.set_index('date')['close']

        # Log Return
        log_ret = np.log(price_series / price_series.shift(1))
        return_series = log_ret.replace([np.inf, -np.inf], np.nan).dropna()

        last_price = float(price_series.iloc[-1])
        last_date = price_series.index[-1]

        # --- [MỚI] KIỂM ĐỊNH TÍNH DỪNG (ADF) ---
        st.markdown("---")
        st.markdown("#### 📉 Kiểm định ADF (Stationarity)")
        try:
            adf_res = check_stationarity(return_series)
            if adf_res['is_stationary']:
                st.success(f"✅ {adf_res['conclusion']}")
            else:
                st.warning(f"⚠️ {adf_res['conclusion']}")
            st.caption(f"ADF Statistic: {adf_res['statistic']:.4f} | p-value: {adf_res['pvalue']:.4f}")
        except Exception as e:
            st.error("Lỗi tính ADF")

        st.markdown("---")
        mode = st.radio("Chế độ:", ["Auto-ARIMA", "Manual"])

        order = None
        use_bic = True  # Mặc định

        if "Manual" in mode:
            p = st.number_input("AR (p)", 0, 10, 1)
            d = st.number_input("I (d)", 0, 2, 0)
            q = st.number_input("MA (q)", 0, 10, 1)
            order = (p, d, q)
        else:
            # [SỬA ĐỔI] Cho phép chọn tiêu chí tối ưu (AIC hoặc BIC)
            criteria = st.radio("Tiêu chí tối ưu mô hình:", ["BIC (Ưu tiên đơn giản)", "AIC (Ưu tiên khớp dữ liệu)"])
            use_bic = True if "BIC" in criteria else False

        st.markdown("---")
        n_steps = st.slider("Dự báo (ngày)", 5, 60, 30)
        run_btn = st.button("🚀 CHẠY DỰ BÁO", type="primary", use_container_width=True)

        st.markdown("---")
        test_size = st.number_input("Backtest size", 10, 90, 30)
        run_backtest_btn = st.button("🔄 Backtest", use_container_width=True)

    with col_results:
        if run_btn:
            try:
                with st.spinner(f"Đang tìm mô hình tối ưu theo {'BIC' if use_bic else 'AIC'}..."):
                    # 1. Fit Model
                    fit_res = fit_arima_on_returns(
                        return_series,
                        order=order,
                        use_bic=use_bic
                    )
                    st.session_state['saved_arima_order'] = fit_res['order']

                    # 2. Dự báo & Tái lập giá
                    fc_df = forecast_arima_returns(fit_res, n_steps, last_price, last_date)

                    if fc_df['forecast_price'].isnull().any():
                        st.error("⚠️ Lỗi tái lập giá: Kết quả dự báo chứa NaN.")
                    else:
                        # 3. Vẽ biểu đồ
                        fig = forecast_figure(price_series, return_series, fc_df,
                                              title=f"Dự báo {ticker} - Model: ARIMA{fit_res['order']}")

                        # [FIX] Lưu ảnh sau khi vẽ xong (trong hàm core đã vẽ rồi)
                        fig.write_image("assets/arima_forecast.png")
                        fc_df.to_csv("assets/arima_forecast_data.csv")
                        st.plotly_chart(fig, use_container_width=True)

                        # 4. Hiển thị Metrics
                        m1, m2, m3 = st.columns(3)

                        # Cột 1: Thông tin Mô hình & AIC/BIC
                        m1.metric("Mô hình", f"ARIMA{fit_res['order']}")
                        m1.caption(f"📉 **AIC:** {fit_res['aic']:.1f} | **BIC:** {fit_res['bic']:.1f}")

                        # Cột 2: Giá mục tiêu
                        end_price_fc = fc_df['forecast_price'].iloc[-1]
                        chg = (end_price_fc - last_price) / last_price * 100
                        m2.metric("Giá mục tiêu", f"{end_price_fc:,.0f}", f"{chg:+.2f}%")

                        # Cột 3: Kiểm định nhiễu trắng
                        wn_status = "✅ Đạt" if fit_res['diagnostics']['is_white_noise'] else "⚠️ Không"
                        m3.metric("White Noise?", wn_status)
                        m3.caption(f"p-value: {fit_res['diagnostics']['ljung_box_pvalue']:.4f}")

                        # 5. Download
                        st.markdown("---")
                        st.download_button(
                            "📥 Tải kết quả (.csv)",
                            fc_df.to_csv().encode('utf-8'),
                            f"arima_{ticker}.csv",
                            "text/csv"
                        )

            except Exception as e:
                st.error(f"❌ Lỗi xử lý: {str(e)}")

        if run_backtest_btn:
            with st.spinner("Đang chạy Backtest..."):

                final_order = order  # Mặc định lấy từ input (None nếu là Auto, hoặc số nếu là Manual)

                # Nếu đang ở chế độ Auto VÀ đã từng chạy dự báo rồi -> Lấy kết quả dự báo ốp vào
                if final_order is None and 'saved_arima_order' in st.session_state:
                    final_order = st.session_state['saved_arima_order']
                    st.caption(f"💡 Đang Backtest trên mô hình cố định: **ARIMA{final_order}** (Lấy từ kết quả Dự báo)")
                # -----------------------------------------------
                bt_res = rolling_backtest(
                    return_series,
                    price_series,
                    test_size=test_size,
                    order=final_order,
                    use_bic=use_bic
                )

                # [FIX LOGIC BACKTEST]
                if "error" in bt_res:
                    st.error(bt_res["error"])
                else:
                    # Chỉ khi không lỗi mới chạy vào đây

                    # 1. Lưu dữ liệu
                    bt_res['plot_data'].to_csv("assets/arima_backtest_data.csv")

                    st.markdown("#### 📊 Hiệu quả Dự báo (Kiểm chứng quá khứ)")

                    # 2. Hiển thị thông tin mô hình
                    st.info(
                        f"ℹ️ Đã Backtest bằng mô hình: **ARIMA{bt_res['order_used']}** (Tối ưu theo **{bt_res.get('criterion', 'Manual')}**)")

                    col_metric1, col_metric2, col_metric3 = st.columns(3)

                    col_metric1.metric("MAPE (Sai số %)", f"{bt_res['mape_pct']:.2f}%")
                    # Lưu ý: Nếu data gốc đơn vị là nghìn đồng, nhân 1000 là đúng. Nếu data gốc là đồng, không cần nhân.
                    # Ở đây giữ nguyên theo code cũ của bạn
                    col_metric1.metric("RMSE (Sai số giá)", f"{bt_res['rmse_vnd'] * 1000:,.0f} VND")
                    col_metric3.metric("MAE (Sai số TB)", f"{bt_res['mae_vnd'] * 1000:,.0f} VND")

                    # 3. Vẽ biểu đồ
                    df_bt = bt_res['plot_data']
                    fig_bt = go.Figure()

                    fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['actual'], name='Thực tế',
                                                line=dict(color='#2962FF')))
                    fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['predicted'], name='Dự báo Model',
                                                line=dict(color='#FF6D00', dash='dot')))
                    fig_bt.update_layout(title=f"Backtest {test_size} phiên gần nhất", height=400,
                                         template="plotly_dark")

                    # [FIX] Lưu ảnh SAU KHI đã add trace
                    fig_bt.write_image("assets/arima_backtest.png")

                    st.plotly_chart(fig_bt, use_container_width=True)

def tab_portfolio_report():
    """Chiến lược: Minh bạch danh mục, Backtest & Chỉ số chuyên sâu"""
    st.header("💼 Chiến lược Đầu tư & Hiệu quả Danh mục")
    import plotly.express as px

    if st.session_state.capm_results is None:
        st.warning("⚠️ Vui lòng quay lại Tab 1 để chạy mô hình CAPM trước.")
        return

    prices = st.session_state.prices_df
    vni = st.session_state.vnindex_df
    capm_res = st.session_state.capm_results

    # --- 1. CẤU HÌNH ---
    st.markdown("### 1. Cấu hình phân nhóm")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        n_q = st.selectbox("Chia nhóm (Quantiles):", [2, 3, 4, 5], index=2, help="Chia thị trường thành N nhóm.")
    with c2:
        w_mode = st.selectbox("Trọng số:", ["equal", "liquidity"])
    with c3:
        st.write("")
        run_bt = st.button("🚀 CHẠY CHIẾN LƯỢC", type="primary", use_container_width=True)

    if run_bt:
        with st.spinner("Đang xử lý dữ liệu..."):
            # --- DEBUG INFO ---
            st.markdown("#### 🕵️ Kiểm tra Dữ liệu (Debug)")
            beta_df = capm_res.copy()
            if 'ticker' in beta_df.columns: beta_df = beta_df.set_index('ticker')

            beta_series = beta_df['beta'].dropna()

            d1, d2 = st.columns(2)
            d1.info(f"Số mã có Beta hợp lệ: **{len(beta_series)}** mã")

            # 2. Phân loại NGAY LẬP TỨC
            try:
                buckets = pd.qcut(beta_series, q=int(n_q), labels=[f"Q{i + 1}" for i in range(int(n_q))])
            except ValueError:
                st.warning("⚠️ Dữ liệu phân bố không đều, chuyển sang chia nhóm thủ công.")
                buckets = pd.Series(index=beta_series.index, data="Unsorted")
                median_b = beta_series.median()
                buckets[beta_series < median_b] = "Q_Low"
                buckets[beta_series >= median_b] = "Q_High"

            # --- [FIX QUAN TRỌNG] ĐẶT TÊN INDEX ĐỂ TRÁNH LỖI KEYERROR ---
            buckets.name = "beta_q"
            buckets.index.name = "ticker"  # <--- DÒNG NÀY SỬA LỖI CỦA BẠN
            # -------------------------------------------------------------

            # --- MINH BẠCH HÓA DANH MỤC ---
            st.markdown("### 2. Chi tiết Danh mục")

            labels = sorted(buckets.unique())
            safe_lab = labels[0]
            risky_lab = labels[-1]

            safe_list = buckets[buckets == safe_lab].index.tolist()
            risky_list = buckets[buckets == risky_lab].index.tolist()

            col_safe, col_risky = st.columns(2)
            with col_safe:
                st.success(f"🛡️ **Danh mục An toàn ({safe_lab})** - Beta TB: {beta_series[safe_list].mean():.2f}")
                st.write(f"**Gồm {len(safe_list)} mã:** {', '.join(safe_list)}")
            with col_risky:
                st.error(f"🚀 **Danh mục Mạo hiểm ({risky_lab})** - Beta TB: {beta_series[risky_list].mean():.2f}")
                st.write(f"**Gồm {len(risky_list)} mã:** {', '.join(risky_list)}")

                # --- 3. BACKTEST (TÍNH TOÁN TRỰC TIẾP & ĐỒNG BỘ DỮ LIỆU) ---
                st.markdown("### 3. Hiệu quả Tăng trưởng (Backtest)")

                # Bước A: Pivot bảng giá từ Session State (Đảm bảo đồng bộ với Tab 1)
                p_pivot = prices.pivot_table(index='date', columns='ticker', values='close')

                # Bước B: Resample về cuối tháng (M) và tính % thay đổi
                # fill_method=None để tránh warning pandas mới
                mret_wide = p_pivot.resample('M').last().pct_change(fill_method=None).dropna(how='all')

                # Bước C: Chuẩn hóa tên (Viết hoa, bỏ khoảng trắng)
                mret_wide.columns = mret_wide.columns.str.strip().str.upper()
                buckets.index = buckets.index.str.strip().str.upper()

                # Hiển thị thông tin Debug
                d2.info(f"Dữ liệu giá tháng: **{mret_wide.shape[0]}** tháng x **{mret_wide.shape[1]}** mã")

                # Bước D: Align dữ liệu (Giao thoa giữa danh sách Beta và danh sách Giá)
                common = buckets.index.intersection(mret_wide.columns)

                if len(common) < len(buckets):
                    missing_count = len(buckets) - len(common)
                    missing_tickers = list(set(buckets.index) - set(common))
                    with st.expander(f"⚠️ Cảnh báo: Có {missing_count} mã thiếu dữ liệu giá lịch sử"):
                        st.write(", ".join(missing_tickers))

                # Lọc dữ liệu chuẩn
                valid_buckets = buckets.loc[common]
                valid_mret_wide = mret_wide[common]

                # --- [FIX MERGE DATA] CHUYỂN ĐỔI FORMAT CHO KHỚP CORE ---
                # Chuyển từ Wide (Cột là Ticker) sang Long (Cột Date, Ticker, Return)
                # để hàm backtest_portfolios có thể merge trên cột 'ticker'
                valid_mret_long = valid_mret_wide.stack().reset_index()
                valid_mret_long.columns = ['date', 'ticker', 'ret_m']

                # Xử lý VNINDEX
                v_pivot = vni.set_index('date')['close']
                rm = v_pivot.resample('M').last().pct_change(fill_method=None)

                # Chạy hàm Backtest (Truyền Long Format vào)
                curves = backtest_portfolios(valid_mret_long, valid_buckets, rm, weight_mode=w_mode)

            # --- TÍNH CHỈ SỐ ---
            def calculate_metrics(equity_series, risk_free=0.0):
                ret = equity_series.pct_change().dropna()
                if len(ret) == 0: return 0, 0, 0, 0
                total_ret = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
                ann_vol = ret.std() * np.sqrt(12)
                ann_ret = ret.mean() * 12
                sharpe = (ann_ret - risk_free) / ann_vol if ann_vol != 0 else 0
                roll_max = equity_series.cummax()
                drawdown = (equity_series - roll_max) / roll_max
                max_dd = drawdown.min()
                return total_ret, ann_vol, sharpe, max_dd

            metrics_data = []
            for name, eq in curves.items():
                tot_r, vol, sh, mdd = calculate_metrics(eq)
                metrics_data.append({
                    "Danh mục": name,
                    "Total Return": tot_r,
                    "Volatility (Năm)": vol,
                    "Sharpe Ratio": sh,
                    "Max Drawdown": mdd
                })

            metrics_df = pd.DataFrame(metrics_data).set_index("Danh mục")

            # --- VẼ BIỂU ĐỒ ---
            fig = go.Figure()
            colors = px.colors.qualitative.Plotly

            for i, (name, eq) in enumerate(curves.items()):
                if name == safe_lab:
                    width, color, dash = 4, "#2ecc71", "solid"
                elif name == risky_lab:
                    width, color, dash = 4, "#e74c3c", "solid"
                else:
                    width, color, dash = 1.5, colors[i % len(colors)], "dot"

                fig.add_trace(go.Scatter(
                    x=eq.index, y=eq.values,
                    mode="lines",
                    name=f"{name} (Sharpe: {metrics_df.loc[name, 'Sharpe Ratio']:.2f})",
                    line=dict(width=width, color=color, dash=dash)
                ))

            fig.update_layout(
                xaxis_title="Thời gian", yaxis_title="NAV Base=1.0",
                template="plotly_dark", height=500, hovermode="x unified",
                legend=dict(orientation="h", y=1.02)
            )
            fig.write_image("assets/portfolio_performance.png")  # <--- LƯU ẢNH QUAN TRỌNG NHẤT
            st.plotly_chart(fig, use_container_width=True)

            # --- BẢNG SO SÁNH ---
            st.markdown("### 4. Bảng So sánh Hiệu quả & Rủi ro")
            st.dataframe(
                metrics_df.style.format({
                    "Total Return": "{:+.2%}", "Volatility (Năm)": "{:.2%}",
                    "Sharpe Ratio": "{:.2f}", "Max Drawdown": "{:.2%}"
                }).background_gradient(cmap="RdYlGn", subset=["Total Return", "Sharpe Ratio"])
                .background_gradient(cmap="RdYlGn_r", subset=["Max Drawdown", "Volatility (Năm)"]),
                use_container_width=True
            )
            metrics_df.to_csv("assets/portfolio_metrics.csv")
            metrics_df.to_parquet(CACHE_DIR / "portfolio_summary.parquet")

            # ============================================================
            # 5. KẾT LUẬN & KHUYẾN NGHỊ (AUTO-INSIGHTS)
            # ============================================================
            st.markdown("---")
            st.markdown("### 5. 🧠 Tổng kết & Khuyến nghị Đầu tư")

            try:
                # 1. Lấy dữ liệu
                safe_stats = metrics_df.loc[safe_lab]
                risky_stats = metrics_df.loc[risky_lab]
                diff_sharpe = risky_stats['Sharpe Ratio'] - safe_stats['Sharpe Ratio']

                # 2. Logic Quyết định
                if diff_sharpe > 0:
                    rec_title = "KHUYẾN NGHỊ: TẤN CÔNG (AGGRESSIVE)"
                    rec_msg = (
                        f"Dựa trên dữ liệu quá khứ, Danh mục **Mạo hiểm ({risky_lab})** đang sử dụng vốn hiệu quả hơn "
                        f"(Sharpe cao hơn {diff_sharpe:.2f}). \n"
                        f"- Lợi nhuận: {risky_stats['Total Return']:.1%}\n"
                        f"- Rủi ro MaxDD: {risky_stats['Max Drawdown']:.1%}\n"
                        f"-> Khuyến nghị: Phân bổ tỷ trọng lớn vào nhóm Beta cao để tối ưu lợi nhuận."
                    )
                    rec_color = "green"
                    winner_list = risky_list
                    loser_list = safe_list
                    winner_name = f"Nhóm Mạo hiểm ({risky_lab})"
                    loser_name = f"Nhóm An toàn ({safe_lab})"
                else:
                    rec_title = "KHUYẾN NGHỊ: PHÒNG THỦ (DEFENSIVE)"
                    rec_msg = (
                        f"Dựa trên dữ liệu quá khứ, Danh mục **An toàn ({safe_lab})** có hiệu suất điều chỉnh rủi ro tốt hơn. "
                        f"Việc chấp nhận thêm rủi ro ở nhóm Beta cao không mang lại lợi nhuận tương xứng.\n"
                        f"- Lợi nhuận: {safe_stats['Total Return']:.1%}\n"
                        f"- Rủi ro MaxDD: {safe_stats['Max Drawdown']:.1%}\n"
                        f"-> Khuyến nghị: Ưu tiên nhóm cổ phiếu Beta thấp để bảo toàn vốn."
                    )
                    rec_color = "blue"
                    winner_list = safe_list
                    loser_list = risky_list
                    winner_name = f"Nhóm An toàn ({safe_lab})"
                    loser_name = f"Nhóm Mạo hiểm ({risky_lab})"

                # Lưu kết luận vào session để dùng cho báo cáo PDF
                st.session_state[
                    'portfolio_conclusion'] = f"{rec_title}\n\n{rec_msg}\n\nDANH SÁCH KHUYẾN NGHỊ ({len(winner_list)} mã):\n{', '.join(winner_list)}"

                # 3. UI Hiển thị (Dashboard Cards)
                if rec_color == "green":
                    st.success(f"## 🚀 {rec_title}\n{rec_msg}")
                else:
                    st.info(f"## 🛡️ {rec_title}\n{rec_msg}")

                # 4. So sánh chi tiết & Danh mục (Giữ nguyên layout đẹp)
                k1, k2, k3 = st.columns(3)
                ret_diff = risky_stats['Total Return'] - safe_stats['Total Return']
                k1.metric("Chênh lệch Lợi nhuận", f"{ret_diff:.1%}", delta_color="normal")

                dd_diff = abs(risky_stats['Max Drawdown']) - abs(safe_stats['Max Drawdown'])
                k2.metric("Chênh lệch Rủi ro (DD)", f"{dd_diff:.1%}", delta_color="inverse")

                k3.metric("Chênh lệch Sharpe", f"{diff_sharpe:.2f}", delta_color="normal")

                # Danh mục (2 Cột)
                st.markdown("#### 📋 Chi tiết Danh mục")
                c_win, c_lose = st.columns(2)
                with c_win:
                    with st.container(border=True):
                        st.markdown(f"### ✅ {winner_name} - ƯU TIÊN")
                        st.success(", ".join(winner_list))
                with c_lose:
                    with st.container(border=True):
                        st.markdown(f"### ⚠️ {loser_name} - CÂN NHẮC")
                        st.code(", ".join(loser_list), language="text")

            except Exception as e:
                st.error(f"Không thể tạo kết luận tự động: {e}")

    # ============================================================
    # 6. XUẤT BÁO CÁO PDF (FINAL FIX - SESSION STATE)
    # ============================================================
    st.markdown("---")
    st.header("🖨️ Xuất Báo cáo Tổng hợp (PDF)")

    # Container cấu hình
    with st.container(border=True):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.info("ℹ️ Báo cáo sẽ bao gồm:\n- EDA\n- CAPM\n- ARIMA\n- Portfolio & Khuyến nghị")
        with c2:
            pdf_title = st.text_input("Tiêu đề báo cáo:", value="Báo cáo Chiến lược Đầu tư VN30", key="pdf_title")

            # Lấy khuyến nghị tự động làm mặc định
            default_note = st.session_state.get('portfolio_conclusion', "Chưa có dữ liệu khuyến nghị.")
            pdf_note = st.text_area("Ghi chú thêm:", value=default_note, height=100, key="pdf_note")

        st.markdown("---")

        # --- LOGIC TẠO VÀ TẢI (TÁCH BIỆT) ---

        # 1. Nút Tạo Báo Cáo
        if st.button("⚙️ KHỞI TẠO PDF", type="primary", use_container_width=True):
            # Kiểm tra dữ liệu
            capm_path = CACHE_DIR / "capm_results_app.parquet"
            port_path = CACHE_DIR / "portfolio_summary.parquet"

            if not (capm_path.exists() and port_path.exists()):
                st.error("⚠️ Thiếu dữ liệu! Vui lòng chạy Tab 1 (CAPM) và Tab 2 (Chiến lược) trước.")
            else:
                with st.spinner("Đang xử lý văn bản và biểu đồ..."):
                    try:
                        # Gọi hàm tạo PDF (trả về file object đang mở)
                        pdf_file_obj = build_pdf_report(
                            title=pdf_title,
                            intro_note=pdf_note,
                            capm_path=str(capm_path),
                            port_summary_path=str(port_path),
                            asset_globs=["assets/*.png"]
                        )

                        # QUAN TRỌNG: Đọc toàn bộ nội dung file vào bộ nhớ đệm (Session State)
                        # Điều này giúp dữ liệu tồn tại vĩnh viễn kể cả khi trang web load lại
                        pdf_file_obj.seek(0)
                        pdf_bytes = pdf_file_obj.read()

                        # Lưu vào Session State
                        st.session_state['pdf_bytes_data'] = pdf_bytes
                        st.session_state[
                            'pdf_filename'] = f"VN30_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"

                        st.success("✅ Đã tạo xong! Nhấn nút tải bên dưới.")

                    except Exception as e:
                        st.error(f"Lỗi tạo PDF: {e}")

        # 2. Nút Tải Xuống (Luôn hiển thị nếu đã có dữ liệu trong Session)
        if 'pdf_bytes_data' in st.session_state:
            st.download_button(
                label=f"📥 TẢI XUỐNG: {st.session_state['pdf_filename']}",
                data=st.session_state['pdf_bytes_data'],
                file_name=st.session_state['pdf_filename'],
                mime="application/pdf",
                use_container_width=True,
                key="btn_final_download"
            )


def main():
    st.title("📊 VN30 QUANTITATIVE DASHBOARD")
    st.markdown("---")

    # 1. Sidebar
    sidebar_data_loading()

    # 2. Check Data
    if st.session_state.prices_df is None:
        st.info("👋 Chào mừng! Vui lòng bấm nút **'Tải Dữ Liệu'** màu đỏ bên trái để bắt đầu.")
        return

    # 3. TABS LAYOUT
    tab1, tab2 = st.tabs(["📈 PHÂN TÍCH & DỰ BÁO", "💼 CHIẾN LƯỢC & BÁO CÁO"])

    # --- TAB 1: TRẢI PHẲNG (FLAT DESIGN) ---
    with tab1:
        # Phần 1: EDA
        tab_eda()

        st.markdown("---")

        # Phần 2: CAPM
        st.header("2. Đánh giá Rủi ro (CAPM)")
        # Gọi trực tiếp hàm logic cũ, nhưng hiển thị phẳng
        tab_capm()

        st.markdown("---")

        # Phần 3: ARIMA
        st.header("3. Dự báo (ARIMA)")
        tab_arima()

        # --- TAB 2: CHIẾN LƯỢC ---
    with tab2:
        # Đã được sửa để hiện minh bạch danh mục
        tab_portfolio_report()


if __name__ == "__main__":
    main()