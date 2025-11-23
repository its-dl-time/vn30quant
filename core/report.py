# FILE: core/report.py
import pandas as pd
from fpdf import FPDF
import glob
import os
import urllib.request

# 1. Cấu hình Font chữ (Tự động tải 3 loại: Thường, Đậm, Nghiêng)
FONT_URL = "https://github.com/google/fonts/raw/main/ofl/robotocondensed/RobotoCondensed-Regular.ttf"
FONT_BOLD_URL = "https://github.com/google/fonts/raw/main/ofl/robotocondensed/RobotoCondensed-Bold.ttf"
FONT_ITALIC_URL = "https://github.com/google/fonts/raw/main/ofl/robotocondensed/RobotoCondensed-Italic.ttf"

FONT_DIR = "assets"
FONT_PATH = os.path.join(FONT_DIR, "Roboto-Regular.ttf")
FONT_BOLD_PATH = os.path.join(FONT_DIR, "Roboto-Bold.ttf")
FONT_ITALIC_PATH = os.path.join(FONT_DIR, "Roboto-Italic.ttf")

def ensure_fonts():
    """Đảm bảo có font chữ hỗ trợ tiếng Việt (Đủ 3 style)"""
    if not os.path.exists(FONT_DIR):
        os.makedirs(FONT_DIR)

    if not os.path.exists(FONT_PATH):
        try:
            print("Đang tải font Regular...")
            urllib.request.urlretrieve(FONT_URL, FONT_PATH)
        except: pass

    if not os.path.exists(FONT_BOLD_PATH):
        try:
            urllib.request.urlretrieve(FONT_BOLD_URL, FONT_BOLD_PATH)
        except: pass

    if not os.path.exists(FONT_ITALIC_PATH):
        try:
            urllib.request.urlretrieve(FONT_ITALIC_URL, FONT_ITALIC_PATH)
        except: pass

class PDFReport(FPDF):
    def __init__(self, title):
        super().__init__()
        ensure_fonts()
        # Đăng ký font Unicode (Regular, Bold, Italic)
        self.add_font('Roboto', '', FONT_PATH, uni=True)
        self.add_font('Roboto', 'B', FONT_BOLD_PATH, uni=True)
        self.add_font('Roboto', 'I', FONT_ITALIC_PATH, uni=True) # Cần cái này để tránh lỗi khi add_image_section

        self.report_title = title

    def header(self):
        self.set_font('Roboto', 'B', 10)
        # Bỏ author, chỉ hiện Title và căn phải
        self.cell(0, 10, f'{self.report_title}', 0, 1, 'R')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Roboto', '', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, label):
        self.set_font('Roboto', 'B', 16)
        self.set_text_color(46, 123, 207) # Xanh dương
        self.cell(0, 10, label, 0, 1, 'L')
        self.ln(4)
        self.set_text_color(0, 0, 0) # Đen lại

    def chapter_body(self, text):
        self.set_font('Roboto', '', 12)
        # Clean text cơ bản
        clean_text = text.replace("**", "").replace("##", "")
        self.multi_cell(0, 8, clean_text)
        self.ln()

    def add_image_section(self, img_path, caption=""):
        if os.path.exists(img_path):
            self.ln(5)
            # Tự động scale ảnh cho vừa trang A4
            self.image(img_path, w=180)
            self.ln(2)
            self.set_font('Roboto', 'I', 9) # Dùng font nghiêng ở đây
            self.cell(0, 5, caption, 0, 1, 'C')
            self.ln(5)

    def add_table(self, df, title="Table Data"):
        self.set_font('Roboto', 'B', 11)
        self.cell(0, 8, title, 0, 1, 'L')

        self.set_font('Roboto', '', 10)
        # Header
        col_width = 190 / len(df.columns)
        self.set_fill_color(240, 240, 240)
        for col in df.columns:
            self.cell(col_width, 8, str(col), 1, 0, 'C', 1)
        self.ln()

        # Rows
        for _, row in df.iterrows():
            for item in row:
                txt = str(item)
                if isinstance(item, float):
                    txt = f"{item:.2f}"
                self.cell(col_width, 8, txt, 1, 0, 'C')
            self.ln()
        self.ln(10)

def build_pdf_report(title, intro_note, capm_path, port_summary_path, asset_globs):
    """
    Hàm chính để tạo báo cáo PDF (Đã bỏ tham số author)
    """
    pdf = PDFReport(title)
    pdf.add_page()

    # 1. Tiêu đề lớn
    pdf.set_font('Roboto', 'B', 24)
    pdf.cell(0, 20, title, 0, 1, 'C')
    pdf.ln(10)

    # 2. Ghi chú / Khuyến nghị
    pdf.chapter_title("1. TỔNG KẾT & KHUYẾN NGHỊ")
    clean_note = intro_note.replace("**", "").replace("##", "")
    pdf.chapter_body(clean_note)

    # 3. Bảng CAPM
    try:
        capm_df = pd.read_parquet(capm_path)
        pdf.chapter_title("2. PHÂN TÍCH RỦI RO (CAPM)")

        # Top Safe
        safe = capm_df.nsmallest(5, 'beta')[['beta', 'alpha', 'R2']]
        pdf.add_table(safe.reset_index(), "Top 5 Mã Phòng thủ (Beta thấp)")

        # Top Risky
        risky = capm_df.nlargest(5, 'beta')[['beta', 'alpha', 'R2']]
        pdf.add_table(risky.reset_index(), "Top 5 Mã Tấn công (Beta cao)")

    except Exception as e:
        pdf.chapter_body(f"Không thể tải dữ liệu CAPM: {e}")

    # 4. Bảng Portfolio Summary
    try:
        pdf.check_page_break(50)
        port_df = pd.read_parquet(port_summary_path)
        pdf.chapter_title("3. HIỆU QUẢ CHIẾN LƯỢC (BACKTEST)")
        pdf.add_table(port_df.reset_index(), "Kết quả Backtest các Danh mục")
    except Exception as e:
        pdf.chapter_body(f"Không thể tải dữ liệu Portfolio: {e}")

    # 5. Chèn Biểu đồ (Assets)
    pdf.add_page()
    pdf.chapter_title("4. BIỂU ĐỒ PHÂN TÍCH")

    import glob
    for g in asset_globs:
        for img in glob.glob(g):
            pdf.add_image_section(img, caption=os.path.basename(img))

    # Output
    output_path = "VN30_Report_Final.pdf"
    pdf.output(output_path)

    # Trả về object file
    return open(output_path, 'rb')