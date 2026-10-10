"""
Bộ kiểm thử tự động khử trùng lặp Task D02 

Chạy: pytest tests/test_deduplication.py -v

Yêu cầu:
    - data/interim/news_articles_validated.csv đã tồn tại
    - Pipeline đã chạy: python -m src.preprocessing.deduplicate
"""
from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

VALIDATED_CSV = Path(os.getenv(
    "VALIDATED_CSV",
    ROOT / "data" / "interim" / "news_articles_validated.csv",
))
DEDUP_CSV = Path(os.getenv(
    "DEDUP_CSV",
    ROOT / "data" / "interim" / "news_articles_deduplicated.csv",
))
REMOVED_CSV = ROOT / "results" / "dedup_data" / "dedup_removed.csv"
CANDIDATES_CSV = ROOT / "results" / "dedup_data" / "dedup_candidates_review.csv"
BA_CSV = ROOT / "results" / "dedup_data" / "dedup_before_after.csv"
POLICY_MD = (
    ROOT / "results" / "dedup_data" / "dedup_policy.md"
    if (ROOT / "results" / "dedup_data" / "dedup_policy.md").exists()
    else ROOT / "docs" / "dedup_policy.md"
)

EXPECTED_COLS = [
    "article_id", "title", "content", "published_at",
    "source", "url", "category_gold",
]


def _normalize_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    norm = unicodedata.normalize("NFC", text).lower()
    norm = re.sub(r"[^\w\s]", " ", norm, flags=re.UNICODE)
    return re.sub(r"\s+", " ", norm).strip()


def _normalize_url(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        return ""
    try:
        p = urlparse(url)
    except Exception:
        return url.strip().lower()
    netloc = p.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return f"{netloc}{p.path.rstrip('/')}"


def _hash_content(text: str) -> str:
    return hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def validated_df() -> pd.DataFrame:
    assert VALIDATED_CSV.exists(), f"Không tìm thấy {VALIDATED_CSV}"
    return pd.read_csv(VALIDATED_CSV, dtype=str)


@pytest.fixture(scope="module")
def dedup_df() -> pd.DataFrame:
    assert DEDUP_CSV.exists(), (
        f"Chưa tạo {DEDUP_CSV}. Chạy: python -m src.preprocessing.deduplicate"
    )
    return pd.read_csv(DEDUP_CSV, dtype=str)


@pytest.fixture(scope="module")
def removed_df() -> pd.DataFrame:
    assert REMOVED_CSV.exists(), f"Không tìm thấy {REMOVED_CSV}"
    return pd.read_csv(REMOVED_CSV, dtype=str)


@pytest.fixture(scope="module")
def candidates_df() -> pd.DataFrame:
    assert CANDIDATES_CSV.exists(), f"Không tìm thấy {CANDIDATES_CSV}"
    return pd.read_csv(CANDIDATES_CSV)


@pytest.fixture(scope="module")
def ba_df() -> pd.DataFrame:
    assert BA_CSV.exists(), f"Không tìm thấy {BA_CSV}"
    return pd.read_csv(BA_CSV)


# 5 Test cases bắt buộc theo đặc tả kỹ thuật
def test_row_count_integrity(validated_df, dedup_df, removed_df):
    """Số dòng sau dedup = trước dedup trừ số removed_article_id duy nhất."""
    n_before = len(validated_df)
    n_after = len(dedup_df)
    n_removed = removed_df["removed_article_id"].nunique()
    assert n_after == n_before - n_removed, (
        f"{n_before} - {n_removed} = {n_before - n_removed} != {n_after}"
    )


def test_no_exact_duplicates_remaining(dedup_df):
    """Tập sạch không còn exact duplicate trên URL, hash content hay norm title."""
    urls = dedup_df["url"].apply(_normalize_url)
    n_url_dup = urls.duplicated().sum()
    assert n_url_dup == 0, f"Còn {n_url_dup} URL trùng lặp"

    hashes = dedup_df["content"].apply(_hash_content)
    n_hash_dup = hashes.duplicated().sum()
    assert n_hash_dup == 0, f"Còn {n_hash_dup} content hash trùng lặp"

    norm_titles = dedup_df["title"].apply(_normalize_text)
    n_title_dup = norm_titles.duplicated().sum()
    assert n_title_dup == 0, f"Còn {n_title_dup} normalized title trùng lặp"


def test_balanced_removal_rate(ba_df):
    """Không source/category nào bị xóa quá 50%."""
    problem = ba_df[ba_df["removal_rate_pct"] > 50]
    assert problem.empty, f"Tỷ lệ xóa bất thường (> 50%):\n{problem.to_string()}"


def test_review_sample_size(candidates_df):
    """File candidates có đủ mẫu remove và keep_both để nghiệm thu."""
    total = len(candidates_df)
    assert total > 0, "File candidates rỗng"

    n_remove = (candidates_df["decision"] == "remove").sum()
    n_keep_both = (candidates_df["decision"] == "keep_both").sum()

    if total >= 60:
        assert n_remove >= 30, f"Cần >= 30 cặp 'remove', chỉ có {n_remove}"
        assert n_keep_both >= 30, f"Cần >= 30 cặp 'keep_both', chỉ có {n_keep_both}"


def test_determinism_and_sorting(dedup_df):
    """Pipeline deterministic 100% và article_id sắp xếp tăng dần."""
    ids = dedup_df["article_id"].tolist()
    assert ids == sorted(ids), "article_id chưa sắp xếp tăng dần"

    check_files = [DEDUP_CSV, REMOVED_CSV, CANDIDATES_CSV, BA_CSV]
    hashes_before = {}
    for f in check_files:
        if f.exists():
            hashes_before[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()

    from src.preprocessing.deduplicate import run_pipeline
    run_pipeline()

    for f in check_files:
        h_after = hashlib.sha256(f.read_bytes()).hexdigest()
        assert h_after == hashes_before.get(f.name), (
            f"File {f.name} khác sau khi chạy lại -> không deterministic"
        )


# Các test case bổ sung về schema và logic
def test_deliverables_exist():
    """Tất cả 5 deliverables đều tồn tại."""
    files = [DEDUP_CSV, REMOVED_CSV, CANDIDATES_CSV, BA_CSV, POLICY_MD]
    for f in files:
        assert f.exists(), f"Deliverable thiếu: {f.relative_to(ROOT)}"


def test_dedup_schema(dedup_df):
    """Tập sạch giữ nguyên 7 cột gốc đúng thứ tự."""
    assert list(dedup_df.columns) == EXPECTED_COLS


def test_removed_schema(removed_df):
    """File removed có đúng 4 cột bắt buộc."""
    expected = ["kept_article_id", "removed_article_id", "duplicate_type", "evidence"]
    assert list(removed_df.columns) == expected


def test_candidates_schema(candidates_df):
    """File candidates có đúng 6 cột bắt buộc."""
    expected = [
        "article_id_1", "article_id_2", "title_similarity",
        "content_similarity", "decision", "reason",
    ]
    assert list(candidates_df.columns) == expected


def test_before_after_schema(ba_df):
    """File before_after có đúng 6 cột bắt buộc."""
    expected = [
        "source", "category_gold", "count_before",
        "count_after", "removed_count", "removal_rate_pct",
    ]
    assert list(ba_df.columns) == expected


def test_transitive_closure():
    """Union-Find: A~B, B~C -> cùng cluster, chỉ giữ 1."""
    from src.preprocessing.deduplicate import UnionFind

    uf = UnionFind()
    uf.union("A", "B")
    uf.union("B", "C")
    clusters = uf.clusters()

    assert uf.find("A") == uf.find("B") == uf.find("C")
    assert len(clusters) == 1
    (members,) = clusters.values()
    assert members == {"A", "B", "C"}


def test_no_false_positive_across_categories(dedup_df):
    """Bài từ chuyên mục khác nhau vẫn còn trong tập sạch."""
    sports = dedup_df[dedup_df["category_gold"] == "Thể thao"]
    edu = dedup_df[dedup_df["category_gold"] == "Giáo dục"]
    assert len(sports) > 0, "Toàn bộ bài Thể thao bị loại - false positive!"
    assert len(edu) > 0, "Toàn bộ bài Giáo dục bị loại - false positive!"


def test_article_id_uniqueness(dedup_df):
    """article_id trong tập sạch phải duy nhất."""
    assert dedup_df["article_id"].duplicated().sum() == 0


def test_removed_not_in_dedup(dedup_df, removed_df):
    """Bài bị loại không xuất hiện trong tập sạch."""
    removed_ids = set(removed_df["removed_article_id"])
    dedup_ids = set(dedup_df["article_id"])
    overlap = removed_ids & dedup_ids
    assert not overlap, f"Bài bị loại còn trong tập sạch: {overlap}"
