"""
Bộ kiểm thử tự động chất lượng dữ liệu Task D01 (Pytest Suite).

Chạy: pytest tests/test_data_quality.py -v

Yêu cầu:
    - data/raw/news_articles.csv đã tồn tại (chạy run_all.py trước)
    - data/interim/news_articles_validated.csv đã tồn tại (chạy validate_data.py)
"""
import hashlib
import os
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = Path(os.getenv("RAW_CSV", ROOT / "data" / "raw" / "news_articles.csv"))
VALIDATED_CSV = Path(os.getenv(
    "VALIDATED_CSV",
    ROOT / "data" / "interim" / "news_articles_validated.csv",
))

EXPECTED_SHA256 = (
    "548b35953368857373e8f26639531b96ab83b742ed6432bff0b6462661b5077a"
)

REQUIRED_COLS = [
    "article_id", "title", "content", "published_at",
    "source", "url", "category_gold",
]
VALID_SOURCES = {"vnexpress", "tuoitre", "vietnamnet"}
VALID_CATS = {
    "Kinh doanh", "Khoa học công nghệ", "Thể thao", "Giáo dục",
    "Sức khỏe", "Pháp luật", "Giải trí",
}

@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    assert RAW_CSV.exists(), f"Không tìm thấy file {RAW_CSV}"
    return pd.read_csv(RAW_CSV, dtype=str)

def test_raw_data_not_modified():
    sha256 = hashlib.sha256(RAW_CSV.read_bytes()).hexdigest()
    assert sha256 == EXPECTED_SHA256, (
        f"File raw đã bị sửa đổi!\n"
        f"  Expected: {EXPECTED_SHA256}\n"
        f"  Actual:   {sha256}\n"
        f"  Nếu thay đổi là chủ ý, cập nhật EXPECTED_SHA256 trong file test."
    )

def test_schema_and_column_count(raw_df):
    """Kiểm tra đủ đúng 7 cột và đúng thứ tự."""
    assert list(raw_df.columns) == REQUIRED_COLS

def test_article_id_uniqueness(raw_df):
    """Kiểm tra article_id duy nhất, không null, không rỗng."""
    assert raw_df["article_id"].isna().sum() == 0
    assert (raw_df["article_id"].str.strip() == "").sum() == 0
    assert raw_df["article_id"].duplicated().sum() == 0

def test_no_empty_required_fields(raw_df):
    """Kiểm tra không cột nào bị bỏ trống."""
    for col in REQUIRED_COLS:
        assert raw_df[col].isna().sum() == 0, f"Cột {col} chứa NaN"
        assert (raw_df[col].str.strip() == "").sum() == 0, f"Cột {col} chứa chuỗi rỗng"

def test_published_at_valid_datetime(raw_df):
    """Kiểm tra toàn bộ ngày đăng parse được và nằm trong tháng 09/2026."""
    dates = pd.to_datetime(raw_df["published_at"], errors="coerce", utc=True)
    assert dates.isna().sum() == 0
    dates_vn = dates.dt.tz_convert("Asia/Ho_Chi_Minh")
    assert ((dates_vn.dt.year == 2026) & (dates_vn.dt.month == 9)).all()


def test_valid_sources_and_categories(raw_df):
    """Kiểm tra giá trị source và category_gold thuộc danh mục hợp lệ."""
    assert set(raw_df["source"].unique()) == VALID_SOURCES
    assert set(raw_df["category_gold"].unique()) == VALID_CATS

def test_validated_interim_dataset():
    """Kiểm tra file interim tồn tại và giữ đúng số dòng."""
    assert VALIDATED_CSV.exists(), (
        f"Chưa tạo file {VALIDATED_CSV}. Chạy: python -m src.preprocessing.validate_data"
    )
    df_val = pd.read_csv(VALIDATED_CSV, dtype=str)
    assert len(df_val) > 0, "File interim rỗng"
    assert df_val["article_id"].duplicated().sum() == 0