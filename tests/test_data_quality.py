"""
Bộ kiểm thử tự động chất lượng dữ liệu Task D01 (Pytest Suite).
Chạy: pytest tests/test_data_quality.py
"""
import hashlib
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "news_articles.csv"
VALIDATED_CSV = ROOT / "data" / "interim" / "news_articles_validated.csv"

REQUIRED_COLS = ["article_id", "title", "content", "published_at", "source", "url", "category_gold"]
VALID_SOURCES = {"vnexpress", "tuoitre", "vietnamnet"}
VALID_CATS = {"Kinh doanh", "Khoa học công nghệ", "Thể thao", "Giáo dục", "Sức khỏe", "Pháp luật", "Giải trí"}

@pytest.fixture(scope="module")
def raw_df():
    assert RAW_CSV.exists(), "Không tìm thấy file data/raw/news_articles.csv"
    return pd.read_csv(RAW_CSV, dtype=str)

def test_raw_data_not_modified():
    """Kiểm tra tính toàn vẹn dữ liệu gốc bằng mã băm SHA256."""
    sha256 = hashlib.sha256(RAW_CSV.read_bytes()).hexdigest()
    assert sha256.startswith("548b3595"), "File raw đã bị sửa đổi so với snapshot ban đầu!"

def test_schema_and_column_count(raw_df):
    """Kiểm tra đủ đúng 7 cột và đúng thứ tự."""
    assert list(raw_df.columns) == REQUIRED_COLS

def test_article_id_uniqueness(raw_df):
    """Kiểm tra article_id là duy nhất, không null, không rỗng."""
    assert raw_df["article_id"].isna().sum() == 0
    assert (raw_df["article_id"].str.strip() == "").sum() == 0
    assert raw_df["article_id"].duplicated().sum() == 0

def test_no_empty_required_fields(raw_df):
    """Kiểm tra không cột nào bị bỏ trống."""
    for col in REQUIRED_COLS:
        assert raw_df[col].isna().sum() == 0
        assert (raw_df[col].str.strip() == "").sum() == 0

def test_published_at_valid_datetime(raw_df):
    """Kiểm tra toàn bộ ngày đăng parse được và nằm trong tháng 09/2026."""
    dates = pd.to_datetime(raw_df["published_at"], errors="coerce")
    assert dates.isna().sum() == 0
    assert ((dates.dt.year == 2026) & (dates.dt.month == 9)).all()

def test_valid_sources_and_categories(raw_df):
    """Kiểm tra giá trị nguồn báo và chuyên mục thuộc danh mục hợp lệ."""
    assert set(raw_df["source"].unique()) == VALID_SOURCES
    assert set(raw_df["category_gold"].unique()) == VALID_CATS

def test_validated_interim_dataset():
    """Kiểm tra file interim tồn tại và giữ đủ 9,304 bài viết."""
    assert VALIDATED_CSV.exists(), "Chưa tạo file data/interim/news_articles_validated.csv"
    df_val = pd.read_csv(VALIDATED_CSV, dtype=str)
    assert len(df_val) == 9304
    assert df_val["article_id"].duplicated().sum() == 0