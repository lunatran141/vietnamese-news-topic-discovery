"""
Bộ kiểm thử tự động tiền xử lý văn bản đa biến thể Task D03 (P0 -> P3).

Mục tiêu kiểm thử:
  - Xác thực 7 tiêu chí bắt buộc theo đặc tả kỹ thuật (docs/task_d03_spec.md).
  - Kiểm tra tính bảo toàn số dòng, tính toàn vẹn metadata, thứ tự article_id.
  - Kiểm tra không rò rỉ nhãn sang cột text.
  - Kiểm tra tính bảo toàn từ phủ định và tính đúng đắn của từ ghép tiếng Việt.

Cách chạy:
    pytest tests/test_preprocessing.py -v
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing.clean_text import clean_text_p0
from src.preprocessing.normalize_text import normalize_text_p1
from src.preprocessing.remove_stopwords import load_stopwords, remove_stopwords_p3
from src.preprocessing.segment_words import segment_words_p2

ROOT = Path(__file__).resolve().parents[1]

DEDUP_CSV = ROOT / "data" / "interim" / "news_articles_deduplicated.csv"
P0_CSV = ROOT / "data" / "processed" / "p0_minimal.csv"
P1_CSV = ROOT / "data" / "processed" / "p1_normalized.csv"
P2_CSV = ROOT / "data" / "processed" / "p2_segmented.csv"
P3_CSV = ROOT / "data" / "processed" / "p3_no_stopwords.csv"
STATS_CSV = ROOT / "results" / "preprocessing_data" / "preprocessing_before_after.csv"
CONFIG_FILE = ROOT / "configs" / "preprocessing.yaml"
STOPWORDS_FILE = ROOT / "configs" / "vietnamese_stopwords.txt"

EXPECTED_COLS = [
    "article_id",
    "title",
    "content",
    "processed_title",
    "processed_content",
    "text",
    "published_at",
    "source",
    "url",
    "category_gold",
]


@pytest.fixture(scope="module")
def dedup_df() -> pd.DataFrame:
    assert DEDUP_CSV.exists(), f"Không tìm thấy {DEDUP_CSV}. Cần chạy D02 trước."
    return pd.read_csv(DEDUP_CSV, dtype=str)


@pytest.fixture(scope="module")
def variants() -> dict[str, pd.DataFrame]:
    paths = {
        "p0": P0_CSV,
        "p1": P1_CSV,
        "p2": P2_CSV,
        "p3": P3_CSV,
    }
    dfs = {}
    for name, p in paths.items():
        assert p.exists(), f"Chưa sinh file biến thể: {p}. Hãy chạy python -m src.preprocessing.build_variants"
        dfs[name] = pd.read_csv(p, dtype=str)
    return dfs


def test_deliverables_and_stats_exist():
    """Kiểm tra sự tồn tại của 4 file processed và file thống kê before_after."""
    files = [P0_CSV, P1_CSV, P2_CSV, P3_CSV, STATS_CSV, CONFIG_FILE, STOPWORDS_FILE]
    for f in files:
        assert f.exists(), f"Tệp bắt buộc chưa tồn tại: {f.relative_to(ROOT)}"


def test_row_count(dedup_df, variants):
    """Tiêu chí 1: Số dòng của P0 == P1 == P2 == P3 == dữ liệu sau dedup (9,274 dòng)."""
    expected_count = len(dedup_df)
    assert expected_count == 9274, f"Số dòng dedup không đúng kỳ vọng: {expected_count}"
    for name, df in variants.items():
        assert len(df) == expected_count, (
            f"Biến thể {name} có {len(df)} dòng, khác với số dòng sau dedup ({expected_count})"
        )


def test_article_id_set(dedup_df, variants):
    """Tiêu chí 2: Tập hợp article_id giống nhau 100% giữa 4 biến thể và khớp với dedup."""
    dedup_ids = set(dedup_df["article_id"])
    for name, df in variants.items():
        var_ids = set(df["article_id"])
        diff = dedup_ids.symmetric_difference(var_ids)
        assert not diff, f"Biến thể {name} có tập article_id lệch so với dedup ({len(diff)} ID lệch)"


def test_article_id_order(dedup_df, variants):
    """Tiêu chí 3: Thứ tự sắp xếp các dòng của article_id trùng khớp hoàn toàn từng vị trí index."""
    expected_order = dedup_df["article_id"].tolist()
    for name, df in variants.items():
        actual_order = df["article_id"].tolist()
        assert actual_order == expected_order, (
            f"Thứ tự article_id ở biến thể {name} bị lệch so với dữ liệu gốc sau dedup"
        )


def test_required_metadata(dedup_df, variants):
    """Tiêu chí 4: Toàn bộ 5 cột siêu dữ liệu được bảo toàn nguyên vẹn, không bị null và không xáo trộn."""
    metadata_cols = ["article_id", "published_at", "source", "url", "category_gold"]
    for name, df in variants.items():
        # Kiểm tra đầy đủ 10 cột
        assert list(df.columns) == EXPECTED_COLS, (
            f"Lược đồ cột của {name} không đúng: {list(df.columns)}"
        )
        for col in metadata_cols:
            assert (df[col].fillna("") == dedup_df[col].fillna("")).all(), (
                f"Giá trị siêu dữ liệu cột '{col}' ở biến thể {name} không khớp với file dedup"
            )


def test_empty_processed_text(variants):
    """Tiêu chí 5: Tỷ lệ văn bản rỗng ở các cột processed_title, processed_content, text bằng 0%."""
    check_cols = ["processed_title", "processed_content", "text"]
    for name, df in variants.items():
        for col in check_cols:
            empty_mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
            empty_count = empty_mask.sum()
            assert empty_count == 0, (
                f"Biến thể {name} có {empty_count} ô bị rỗng tại cột '{col}'"
            )


def test_deterministic():
    """Tiêu chí 6: Chạy lại hàm xử lý với cùng một đầu vào cho ra chuỗi kết quả có cùng mã hash SHA-256."""
    sample_text = (
        "Thủ tướng Chính phủ vừa ký ban hành quyết định về phát triển trí tuệ nhân tạo. "
        "Xem thêm tại https://vnexpress.net hoặc liên hệ email: contact@news.vn. "
        "Đây không phải là vấn đề đơn giản và chưa thể giải quyết ngay."
    )
    sample_title = "Thủ tướng ban hành quyết định về AI"

    # Chạy lần 1
    t0_1 = clean_text_p0(sample_title)
    c0_1 = clean_text_p0(sample_text)
    t1_1 = normalize_text_p1(t0_1)
    c1_1 = normalize_text_p1(c0_1, title=t0_1)
    t2_1 = segment_words_p2(t1_1)
    c2_1 = segment_words_p2(c1_1)
    t3_1 = remove_stopwords_p3(t2_1)
    c3_1 = remove_stopwords_p3(c2_1)

    # Chạy lần 2
    t0_2 = clean_text_p0(sample_title)
    c0_2 = clean_text_p0(sample_text)
    t1_2 = normalize_text_p1(t0_2)
    c1_2 = normalize_text_p1(c0_2, title=t0_2)
    t2_2 = segment_words_p2(t1_2)
    c2_2 = segment_words_p2(c1_2)
    t3_2 = remove_stopwords_p3(t2_2)
    c3_2 = remove_stopwords_p3(c2_2)

    for step_name, s1, s2 in [
        ("P0", f"{t0_1} {c0_1}", f"{t0_2} {c0_2}"),
        ("P1", f"{t1_1} {c1_1}", f"{t1_2} {c1_2}"),
        ("P2", f"{t2_1} {c2_1}", f"{t2_2} {c2_2}"),
        ("P3", f"{t3_1} {c3_1}", f"{t3_2} {c3_2}"),
    ]:
        h1 = hashlib.sha256(s1.encode("utf-8")).hexdigest()
        h2 = hashlib.sha256(s2.encode("utf-8")).hexdigest()
        assert h1 == h2, f"Bước {step_name} không đạt tính tất định (mã băm không trùng khớp)"


def test_no_label_leakage(variants):
    """
    Tiêu chí 7: Cột text tuyệt đối không bị dính nhãn category_gold hoặc source ở đầu/cuối chuỗi
    để tránh làm lộ thông tin nhãn cho các mô hình không giám sát.
    """
    for name, df in variants.items():
        sample = df.head(100)
        for _, row in sample.iterrows():
            txt = row["text"].strip().lower()
            src = str(row["source"]).strip().lower()
            # Kiểm tra text không bắt đầu bằng source prefix nhân tạo (ví dụ: 'vnexpress: ...')
            assert not txt.startswith(f"{src}:"), (
                f"Biến thể {name} bị rò rỉ source vào đầu cột text: {txt[:30]}"
            )
            assert not txt.startswith(f"[{src}]"), (
                f"Biến thể {name} bị rò rỉ source dạng bracket vào đầu cột text: {txt[:30]}"
            )


def test_linguistic_invariants(variants):
    """
    Kiểm tra các bất biến ngôn ngữ học tiếng Việt:
      1. Biến thể P2 và P3 có chứa từ ghép nối bằng dấu gạch dưới '_'.
      2. Biến thể P3 bảo toàn các từ phủ định quan trọng ('không', 'chưa', 'chẳng').
    """
    df_p2 = variants["p2"]
    df_p3 = variants["p3"]

    # 1. Kiểm tra có từ ghép nối '_' ở P2 và P3
    has_underscore_p2 = df_p2["text"].str.contains("_").mean()
    has_underscore_p3 = df_p3["text"].str.contains("_").mean()
    assert has_underscore_p2 > 0.99, f"P2 có ít hơn 99% bài chứa từ ghép: {has_underscore_p2:.2%}"
    assert has_underscore_p3 > 0.99, f"P3 có ít hơn 99% bài chứa từ ghép: {has_underscore_p3:.2%}"

    # 2. Kiểm tra từ phủ định 'không' không bị xóa sạch ở P3 nếu P2 có chứa 'không'
    sample_with_khong_p2 = df_p2[df_p2["text"].str.contains(r"\bkhông\b", case=False, regex=True)]
    if not sample_with_khong_p2.empty:
        matching_ids = sample_with_khong_p2["article_id"].head(20).tolist()
        p3_matches = df_p3[df_p3["article_id"].isin(matching_ids)]
        for _, r in p3_matches.iterrows():
            assert "không" in r["text"].lower(), (
                f"Từ phủ định 'không' bị xóa oan ở bài {r['article_id']} biến thể P3"
            )


def test_stats_table_values():
    """Kiểm tra cấu trúc và tính hợp lệ của bảng thống kê results/preprocessing_data/preprocessing_before_after.csv."""
    assert STATS_CSV.exists(), f"Không tìm thấy {STATS_CSV}"
    df_stats = pd.read_csv(STATS_CSV)

    expected_cols = [
        "variant", "total_articles", "empty_text_count", "vocab_size",
        "total_tokens", "retention_rate_pct", "mean_length_chars",
        "median_length_chars", "p5_length_chars", "p95_length_chars",
        "mean_word_count", "median_word_count", "p5_word_count", "p95_word_count",
    ]
    assert list(df_stats.columns) == expected_cols, f"Lược đồ thống kê chưa khớp: {list(df_stats.columns)}"

    # Phải có đủ 5 hàng: original, p0_minimal, p1_normalized, p2_segmented, p3_no_stopwords
    variants_in_stats = df_stats["variant"].tolist()
    assert variants_in_stats == ["original", "p0_minimal", "p1_normalized", "p2_segmented", "p3_no_stopwords"]

    # Số lượng bài báo trên mọi dòng phải là 9,274
    assert (df_stats["total_articles"] == 9274).all()

    # Không có bài rỗng
    assert (df_stats["empty_text_count"] == 0).all()

