"""
ARIMA Module - Final "Exam Ready" Version
Features:
1. Manual & Auto Order Selection
2. Price Reconstruction connected seamlessly
3. Real Rolling Backtest
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# 1. UTILS
# ============================================================================

def check_stationarity(series: pd.Series) -> dict:
    series = series.dropna()
    if len(series) < 10:
        return {'statistic': 0, 'pvalue': 1, 'is_stationary': False, 'conclusion': 'Not enough data'}
    result = adfuller(series)
    return {
        'statistic': result[0],
        'pvalue': result[1],
        'is_stationary': result[1] < 0.05,
        'conclusion': "Chuỗi dừng (Stationary)" if result[1] < 0.05 else "Chuỗi không dừng (Non-stationary)"
    }

# ============================================================================
# 2. CORE MODELING (Manual + Auto Support)
# ============================================================================

def fit_arima_on_returns(returns_series: pd.Series, order=None, max_p=5, max_q=5, use_bic=True):
    """
    Fit ARIMA.
    - Nếu `order` != None: Chạy chế độ Manual (User tự chọn p,d,q)
    - Nếu `order` == None: Chạy Auto-ARIMA
    """
    returns = returns_series.dropna()

    # --- CASE 1: MANUAL MODE ---
    if order is not None:
        # Ép kiểu về list/tuple int (p,d,q)
        try:
            model_sm = ARIMA(returns, order=order).fit()
            final_order = order
            # Tính AIC/BIC thủ công
            aic = model_sm.aic
            bic = model_sm.bic
        except Exception as e:
            raise ValueError(f"Không thể chạy mô hình với tham số {order}. Lỗi: {str(e)}")

    # --- CASE 2: AUTO MODE ---
    else:
        ic = 'bic' if use_bic else 'aic'
        model_auto = pm.auto_arima(
            returns,
            start_p=0, max_p=max_p,
            start_q=0, max_q=max_q,
            d=None, max_d=1,
            seasonal=False,
            information_criterion=ic,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
            with_intercept=True
        )
        final_order = model_auto.order
        model_sm = ARIMA(returns, order=final_order).fit()
        aic = model_sm.aic
        bic = model_sm.bic

    # --- DIAGNOSTICS ---
    residuals = model_sm.resid
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    lb_pvalue = lb_test['lb_pvalue'].values[0]
    arch_test = het_arch(residuals)

    diag = {
        'mean': residuals.mean(),
        'std': residuals.std(),
        'ljung_box_pvalue': lb_pvalue,
        'is_white_noise': lb_pvalue > 0.05,
        'arch_pvalue': arch_test[1],
        'has_arch': arch_test[1] < 0.05
    }

    return {
        'model': model_sm,
        'order': final_order,
        'aic': aic,
        'bic': bic,
        'diagnostics': diag
    }

# ============================================================================
# 3. FORECAST & PRICE RECONSTRUCTION
# ============================================================================

def forecast_arima_returns(fit_res, n_steps, last_price, last_date):
    """
    Dự báo Return -> Tái lập Giá nối tiếp từ last_price
    """
    model = fit_res['model']

    # 1. Dự báo Return
    forecast_res = model.get_forecast(steps=n_steps)
    pred_returns = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int(alpha=0.05)

    # 2. Tạo ngày (Business Days)
    next_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_steps)

    df_fc = pd.DataFrame(index=next_dates)

    # Lưu lại return dự báo để vẽ chart dưới
    df_fc['forecast_return'] = pred_returns.values
    df_fc['lower_return'] = conf_int.iloc[:, 0].values
    df_fc['upper_return'] = conf_int.iloc[:, 1].values

    # 3. Tái lập Giá (Cumulative Product)
    # Công thức: P_t = P_last * exp(cumsum(return))

    # Tính tổng tích lũy của return kể từ ngày dự báo đầu tiên
    cumsum_ret = df_fc['forecast_return'].cumsum()
    cumsum_lower = df_fc['lower_return'].cumsum()
    cumsum_upper = df_fc['upper_return'].cumsum()

    # Tính ra giá tuyệt đối (VND)
    df_fc['forecast_price'] = last_price * np.exp(cumsum_ret)
    df_fc['lower_price'] = last_price * np.exp(cumsum_lower)
    df_fc['upper_price'] = last_price * np.exp(cumsum_upper)

    return df_fc
# ============================================================================
# 4. REAL BACKTEST
# ============================================================================

# FILE: core/arima.py

def rolling_backtest(returns_series, prices_series, n_steps=1, test_size=30, order=None, use_bic=True):
    """
    Backtest: Tái lập giá để tính sai số tiền thật (VND)
    Có hỗ trợ chọn AIC/BIC
    """
    # 1. Setup
    returns = returns_series.dropna()
    if len(returns) < test_size + 50:
        return {"error": "Dữ liệu quá ngắn."}

    # 2. Tách Train/Test
    train_len = len(returns) - test_size
    train_returns = returns.iloc[:train_len]
    test_returns = returns.iloc[train_len:]

    # Lấy dữ liệu giá tương ứng
    actual_prices = prices_series.reindex(test_returns.index)
    last_train_price = float(prices_series.loc[train_returns.index[-1]])

    try:
        # 3. Fit Model
        if order:
            # Manual Mode
            model = ARIMA(train_returns, order=order).fit()
        else:
            # Auto Mode: CHỌN AIC HAY BIC DỰA VÀO INPUT
            ic = 'bic' if use_bic else 'aic'

            model_auto = pm.auto_arima(
                train_returns,
                start_p=0, max_p=3,
                d=None, max_d=1,
                information_criterion=ic,  # <--- QUAN TRỌNG: Sửa dòng này
                trace=False,
                error_action='ignore',
                suppress_warnings=True
            )
            model = ARIMA(train_returns, order=model_auto.order).fit()

        # 4. Dự báo Return
        fc_res = model.get_forecast(steps=len(test_returns))
        pred_log_returns = fc_res.predicted_mean

        # 5. Tái lập Giá (Price Reconstruction)
        cumsum_ret = pred_log_returns.cumsum()
        pred_prices = last_train_price * np.exp(cumsum_ret.values)

        # 6. Tính sai số trên GIÁ (VND)
        df_res = pd.DataFrame({
            'actual': actual_prices.values,
            'predicted': pred_prices
        }, index=test_returns.index).dropna()

        # Metrics
        rmse_vnd = np.sqrt(((df_res['actual'] - df_res['predicted']) ** 2).mean())
        mae_vnd = (df_res['actual'] - df_res['predicted']).abs().mean()
        mape_pct = (df_res['actual'] - df_res['predicted']).abs().div(df_res['actual']).mean() * 100

        return {
            'rmse_vnd': rmse_vnd,
            'mae_vnd': mae_vnd,
            'mape_pct': mape_pct,
            'test_size': len(df_res),
            'plot_data': df_res,
            'order_used': model.model.order,  # Trả về tham số mô hình đã dùng
            'criterion': 'BIC' if use_bic else 'AIC'  # Trả về tiêu chí đã dùng
        }

    except Exception as e:
        return {"error": f"Lỗi Backtest: {str(e)}"}

# ============================================================================
# 5. VISUALIZATION
# ============================================================================

def forecast_figure(history_price, history_returns, forecast_df, title="ARIMA Forecast"):
    """
    Vẽ chart với tính năng Interactive: Range Slider & Auto-scale
    """
    # Tạo subplot 2 dòng: Giá (70%) và Return (30%)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f"{title} - Price Prediction", "Return Prediction"),
        row_heights=[0.7, 0.3]
    )

    # Lấy toàn bộ lịch sử (để người dùng tự zoom) hoặc cắt bớt nếu quá dài
    # Ở đây ta lấy tối đa 2 năm gần nhất để load cho nhanh
    hist_p = history_price.tail(500)
    hist_r = history_returns.tail(500)

    # --- CHART 1: PRICE ---
    # Lịch sử
    fig.add_trace(go.Scatter(
        x=hist_p.index, y=hist_p.values,
        mode='lines', name='History',
        line=dict(color='#2962FF', width=1.5)  # Xanh dương đậm
    ), row=1, col=1)

    # Nối mạch (Điểm cuối lịch sử -> Điểm đầu dự báo)
    last_hist_date = hist_p.index[-1]
    last_hist_val = hist_p.iloc[-1]
    x_fc = [last_hist_date] + list(forecast_df.index)
    y_fc = [last_hist_val] + list(forecast_df['forecast_price'])

    # Dự báo
    fig.add_trace(go.Scatter(
        x=x_fc, y=y_fc,
        mode='lines', name='Forecast',
        line=dict(color='#F50057', width=2, dash='dot')  # Hồng đậm
    ), row=1, col=1)

    # Dải tin cậy (Fan Chart)
    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df['upper_price'],
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df['lower_price'],
        mode='lines', line=dict(width=0), fill='tonexty',
        fillcolor='rgba(245, 0, 87, 0.1)', name='95% Confidence'
    ), row=1, col=1)

    # --- CHART 2: RETURN ---
    fig.add_trace(go.Scatter(
        x=hist_r.index, y=hist_r.values,
        mode='lines', name='Hist Return',
        line=dict(color='grey', width=1)
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df['forecast_return'],
        mode='lines', name='Fcst Return',
        line=dict(color='#00C853')  # Xanh lá
    ), row=2, col=1)

    # --- LAYOUT & INTERACTIVITY ---
    fig.update_layout(
        height=650,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Thêm Range Slider và Buttons
    fig.update_xaxes(
        rangeslider_visible=False,  # Tắt slider mặc định ở dưới cùng vì tốn chỗ
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(step="all", label="ALL")
            ])
        ),
        row=1, col=1
    )

    # QUAN TRỌNG: Autoscale trục Y không bắt đầu từ 0
    fig.update_yaxes(autorange=True, fixedrange=False, row=1, col=1)
    fig.update_yaxes(autorange=True, fixedrange=False, row=2, col=1)

    return fig