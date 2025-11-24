# VN30 QUANTITATIVE ANALYSIS DASHBOARD
### Đồ án môn học: Gói phần mềm ứng dụng cho Tài chính 1

---

## 👨‍🎓 THÔNG TIN SINH VIÊN
* **Họ và tên:** PHẠM MẠNH QUYỀN
* **MSSV:** K244141694
* **Giảng viên hướng dẫn:** THS. NGÔ PHÚ THANH

---

Dự án này xây dựng một dashboard định lượng cho VN30 bằng Python và Streamlit, tập trung vào:

Phân tích dữ liệu giá & lợi suất VN30

Dự báo ARIMA cho cổ phiếu đơn lẻ (ví dụ GAS)

Ước lượng CAPM (α, β, R²) với VNINDEX làm thị trường

Xây dựng danh mục theo beta quantile (Q1–Q4) và backtest hiệu quả so với thị trường

Xuất báo cáo PDF tự động phục vụ bài thi / báo cáo học phần


1. Cấu trúc dự án

Cấu trúc cơ bản (tên thư mục có thể khác chút tuỳ máy bạn):

.
├─ app.py                 # File chính chạy Streamlit
├─ core/
│  ├─ data_io.py          # Lấy dữ liệu từ API / CSV, cache dữ liệu, load RF
│  ├─ clean.py            # Làm sạch dữ liệu, tính log-return, winsorize
│  ├─ eda.py              # Hàm EDA: summary stats, histogram, heatmap, v.v.
│  ├─ arima.py            # Fit ARIMA, auto_arima, backtest, forecast
│  ├─ capm.py             # Chạy CAPM, tính alpha, beta, R², CI95, p-value
│  ├─ portfolio.py        # Chia beta thành Q1–Q4, backtest, tính NAV & metrics
│  ├─ report.py           # Build báo cáo (PDF / HTML) từ kết quả mô hình
│  └─ __init__.py
├─ data/
│  ├─ cache/              # Cache dữ liệu giá từ CafeF / API khác (CSV)
│  ├─ raw/                # (Tuỳ chọn) CSV tải tay
│  └─ rf/                 # CSV lãi suất TPCP 1Y (risk-free)
├─ outputs/
│  ├─ figures/            # Biểu đồ EDA, ARIMA, CAPM, Portfolio
│  └─ reports/            # Báo cáo PDF xuất ra từ dashboard
└─ requirements.txt       # Danh sách thư viện Python


🔎 Khi không chắc, mở từng file core/*.py để xem đường dẫn chính xác tới thư mục dữ liệu (cache_dir, rf_path, …).

2. Yêu cầu hệ thống & cài đặt

2.1. Yêu cầu

Python 3.9+

pip, virtualenv (khuyến nghị)

Kết nối Internet (lần chạy đầu để tải dữ liệu từ CafeF / API / Investing)

2.2. Cài đặt

# 1. Tạo và kích hoạt virtualenv (tuỳ OS)

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 2. Cài thư viện
pip install -r requirements.txt

3. Lần chạy đầu tiên cần làm gì?
3.1. Chuẩn bị dữ liệu RF (risk-free) - ĐÃ CÓ SẴN, KIỂM TRA TRONG data_cache, NẾU KHÔNG CÓ THÌ:

Tải dữ liệu lợi suất TPCP Việt Nam kỳ hạn 1 năm (Vietnam 1Y bond) từ Investing hoặc nguồn tương đương.

Lưu file CSV vào thư mục, ví dụ:

data/rf/vn_1y_bond.csv

Đảm bảo data_io.py hoặc clean.py trỏ đúng đường dẫn file RF này (xem hàm kiểu load_rf_investing_csv()).

3.2. Chạy app lần đầu

streamlit run app.py


Lần đầu, nếu chọn nguồn dữ liệu: CafeF API, app sẽ tải dữ liệu giá VN30 + VNINDEX và lưu vào data/cache/.

Thời gian chạy phụ thuộc tốc độ mạng và số mã: thường mất 5 phút cho full VN30 từ 2020–2025.

Các lần sau có thể chọn Load từ Cache để chạy nhanh hơn.

4. Các tab & pipeline phân tích

4.1. Tab EDA / Dữ liệu

Chọn khoảng thời gian (mặc định 2020–nay).

Chọn rổ cổ phiếu (VN30 hoặc subset).

App sẽ:

Tải/đọc dữ liệu → data_io.py

Làm sạch + tính log-return → clean.py

Tạo:

Thống kê mô tả (mean, std, skew, kurt)

Histogram, boxplot, line chart

Heatmap tương quan VN30

Dữ liệu đã xử lý được lưu dưới dạng DataFrame và/hoặc CSV (ví dụ prices_clean.csv, returns.csv).

4.2. Tab ARIMA

Chọn 1 mã cổ phiếu (ví dụ GAS).

Chọn tham số ARIMA hoặc để auto (AIC/BIC).

arima.py sẽ:

Kiểm định ADF (tính dừng)

Chạy auto_arima → đề xuất (p,d,q) tối ưu theo AIC/BIC

Fit lại bằng statsmodels

Backtest rolling, tính MAPE, RMSE, MAE

Tái lập giá từ forecast return (tích lũy mũ).

Tab hiển thị:

Biểu đồ giá + forecast

Biểu đồ return + forecast

Bảng lỗi backtest (AIC vs BIC models)

4.3. Tab CAPM

Sử dụng dữ liệu tháng:

Lợi suất tháng từng mã VN30

Lợi suất tháng VNINDEX

Lãi suất RF tháng (từ TPCP 1Y)

capm.py:

Chạy hồi quy OLS: CAPM

Tính α, β, R², p-value, CI95, n

Tab hiển thị:

Bảng CAPM results (có export CSV)

Biểu đồ Beta bar chart

Scatter Beta–R², Alpha–Beta

4.4. Tab Portfolio / Backtest

portfolio.py:

Lấy bảng CAPM → phân nhóm beta_q ∈ {1,2,3,4} bằng qcut.

Tạo danh mục Q1–Q4 (equal-weight).

Tính return danh mục theo tháng:

Tái lập NAV từ log-return:

Tính:

Lợi suất năm hóa

Vol năm hóa

Sharpe

Max Drawdown

Tab hiển thị:

Đường NAV Q1–Q4 vs MKT

Bảng portfolio_metrics

Card khuyến nghị: Aggressive / Balanced / Conservative

4.5. Tab Report / Export

report.py gom toàn bộ kết quả (EDA, ARIMA, CAPM, Q1–Q4, khuyến nghị)

Xuất file PDF/HTML trong outputs/reports/.

5. Tuỳ chọn & cấu hình

Trong sidebar app, bạn có thể:

Chọn Nguồn dữ liệu:

CafeF API

Cache (CSV đã lưu)

Chọn Khoảng thời gian: 2020–nay, hoặc custom.

Chọn Rổ cổ phiếu: VN30 hoặc subset.

Chọn Risk-free:

0%

TPCP 1Y (CSV)

Chọn Mô hình ARIMA:

Auto AIC

Auto BIC

Manual (p,d,q)

6. Lưu ý & Hạn chế

Không tính phí giao dịch, thuế, trượt giá → backtest có thể lạc quan hơn thực tế.

Mặc định dùng equal-weight, chưa tối ưu mean–variance.

CAPM là mô hình đơn nhân tố, chưa xét size, value, momentum.

Chất lượng dữ liệu phụ thuộc vào API (CafeF, Investing). Hãy kiểm tra cẩn thận khi dùng cho mục đích thực tế.

7. Bản quyền & Mục đích

Dự án được xây dựng cho mục đích học tập, nghiên cứu và bài thi cuối kỳ.

Không phải khuyến nghị đầu tư chính thức.

Bạn có thể fork, chỉnh sửa, mở rộng (thêm multi-factor, GARCH, machine learning, v.v.) tuỳ nhu cầu.