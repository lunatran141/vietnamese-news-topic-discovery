"""
Script thẩm định dữ liệu thô (Task D01 Data Validation Pipeline).
Chạy độc lập: python -m src.preprocessing.validate_data
"""
import hashlib
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "raw" / "news_articles.csv"
OUTPUT_CSV = ROOT / "data" / "interim" / "news_articles_validated.csv"

REQUIRED_COLS = ["article_id", "title", "content", "published_at", "source", "url", "category_gold"]
VALID_SOURCES = {"vnexpress", "tuoitre", "vietnamnet"}
VALID_CATS = {"Kinh doanh", "Khoa học công nghệ", "Thể thao", "Giáo dục", "Sức khỏe", "Pháp luật", "Giải trí"}

def validate_raw_data() -> pd.DataFrame:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu gốc tại: {RAW_CSV}")

    # Đọc dữ liệu (chế độ chỉ đọc)
    df = pd.read_csv(RAW_CSV, dtype={c: str for c in ["article_id", "url", "title", "content", "source", "category_gold"]}, low_memory=False)

    # 1. Kiểm tra đủ đúng 7 cột
    assert list(df.columns) == REQUIRED_COLS, f"Sai schema cột! Hiện tại: {list(df.columns)}"

    # 2. article_id duy nhất và không rỗng
    assert df["article_id"].isna().sum() == 0, "article_id chứa giá trị rỗng (NaN)!"
    assert (df["article_id"].str.strip() == "").sum() == 0, "article_id chứa chuỗi trắng!"
    assert df["article_id"].duplicated().sum() == 0, "article_id bị trùng lặp!"

    # 3. Không thiếu trường bắt buộc
    for col in ["title", "content", "source", "url", "category_gold"]:
        assert df[col].isna().sum() == 0, f"Cột {col} chứa giá trị NaN!"
        assert (df[col].str.strip() == "").sum() == 0, f"Cột {col} chứa chuỗi rỗng!"

    # 4. Ngày đăng parse hợp lệ và thuộc tháng 09/2026
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    assert df["published_at"].isna().sum() == 0, "Có bài viết không thể parse ngày đăng!"
    assert ((df["published_at"].dt.year == 2026) & (df["published_at"].dt.month == 9)).all(), "Tồn tại bài viết ngoài tháng 09/2026!"

    # 5. Nguồn báo và chuyên mục thuộc tập hợp cho phép
    assert set(df["source"].unique()).issubset(VALID_SOURCES), f"Nguồn báo không hợp lệ: {set(df['source'].unique()) - VALID_SOURCES}"
    assert set(df["category_gold"].unique()).issubset(VALID_CATS), f"Chuyên mục không hợp lệ: {set(df['category_gold'].unique()) - VALID_CATS}"

    # 6. Giữ nguyên đúng 9,304 bài viết
    assert len(df) == 9304, f"Số lượng bài viết khác 9,304! Hiện có: {len(df)}"

    # Lưu dữ liệu sang interim
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✓ Thẩm định thành công 100%. Đã lưu interim dataset: {OUTPUT_CSV.relative_to(ROOT)}")
    return df

if __name__ == "__main__":
    try:
        validate_raw_data()
    except Exception as e:
        print(f"✗ Thẩm định thất bại: {e}")
        sys.exit(1)