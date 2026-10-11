"""
Kiểm thử tự động cho Task D05: Dataset Splits.
Module: tests/test_splits.py

Xác thực 5 yêu cầu cốt lõi:
1. Tránh rò rỉ dữ liệu giữa các cụm sự kiện / near-duplicate.
2. Bảo toàn tỷ lệ phân tầng của 7 chuyên mục.
3. Sử dụng seed cố định và tỷ lệ chuẩn 70% / 15% / 15%.
4. Chỉ lưu danh sách article_id trong data/splits/, không nhân bản file dữ liệu nặng.
5. Kiểm tra tính toàn vẹn, tính độc lập và độ bao phủ 100% (9,274 bài viết).
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest
import yaml

from src.preprocessing.build_splits import build_event_clusters


SPLIT_DIR = Path("data/splits")
CONFIG_PATH = Path("configs/split.yaml")
DEDUP_PATH = Path("data/interim/news_articles_deduplicated.csv")
CANDIDATES_PATH = Path("results/dedup_data/dedup_candidates_review.csv")


def test_split_files_and_config_exist():
    """Kiểm tra sự tồn tại của 3 file split IDs và file cấu hình YAML."""
    assert CONFIG_PATH.is_file(), f"Thiếu file cấu hình: {CONFIG_PATH}"
    assert (SPLIT_DIR / "train_ids.csv").is_file(), "Thiếu file train_ids.csv"
    assert (SPLIT_DIR / "validation_ids.csv").is_file(), "Thiếu file validation_ids.csv"
    assert (SPLIT_DIR / "test_ids.csv").is_file(), "Thiếu file test_ids.csv"


def test_config_parameters_and_test_lock():
    """Kiểm tra các tham số trong split.yaml và chính sách khóa tập test."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["split"]

    assert cfg["train_ratio"] == 0.70
    assert cfg["val_ratio"] == 0.15
    assert cfg["test_ratio"] == 0.15
    assert cfg["random_seed"] == 42
    assert cfg["stratify_column"] == "category_gold"
    assert cfg["id_column"] == "article_id"

    # Kiểm tra chính sách khóa Test
    test_policy = cfg.get("test_policy", {})
    assert test_policy.get("lock_test_set") is True
    assert test_policy.get("allow_hyperparameter_tuning_on_test") is False
    assert test_policy.get("allow_model_selection_on_test") is False


def test_total_count_and_100_percent_coverage():
    """Kiểm tra tổng số bài là 9,274 và bao phủ 100% dataset sau dedup."""
    df_dedup = pd.read_csv(DEDUP_PATH)
    all_expected_ids = set(df_dedup["article_id"])
    assert len(all_expected_ids) == 9274

    train_ids = set(pd.read_csv(SPLIT_DIR / "train_ids.csv")["article_id"])
    val_ids = set(pd.read_csv(SPLIT_DIR / "validation_ids.csv")["article_id"])
    test_ids = set(pd.read_csv(SPLIT_DIR / "test_ids.csv")["article_id"])

    assert len(train_ids) == 6493
    assert len(val_ids) == 1391
    assert len(test_ids) == 1390

    total_split_ids = train_ids | val_ids | test_ids
    assert len(total_split_ids) == 9274
    assert total_split_ids == all_expected_ids


def test_zero_overlap_strictly_disjoint():
    """Kiểm tra tính rời rạc tuyệt đối: Không có bất kỳ ID nào xuất hiện ở nhiều hơn 1 tập."""
    train_ids = set(pd.read_csv(SPLIT_DIR / "train_ids.csv")["article_id"])
    val_ids = set(pd.read_csv(SPLIT_DIR / "validation_ids.csv")["article_id"])
    test_ids = set(pd.read_csv(SPLIT_DIR / "test_ids.csv")["article_id"])

    assert len(train_ids & val_ids) == 0, "Train và Validation bị trùng ID!"
    assert len(train_ids & test_ids) == 0, "Train và Test bị trùng ID (Rò rỉ dữ liệu)!"
    assert len(val_ids & test_ids) == 0, "Validation và Test bị trùng ID!"


def test_only_article_id_stored_in_split_files():
    """Kiểm tra yêu cầu 4: File trong data/splits/ chỉ chứa article_id."""
    for filename in ["train_ids.csv", "validation_ids.csv", "test_ids.csv"]:
        df = pd.read_csv(SPLIT_DIR / filename)
        assert list(df.columns) == ["article_id"], f"{filename} chứa các cột thừa ngoài article_id: {list(df.columns)}"
        assert df["article_id"].isna().sum() == 0
        assert (df["article_id"].str.strip() == "").sum() == 0


def test_stratification_parity_category_gold():
    """Kiểm tra tỷ lệ phân tầng của 7 chuyên mục: Độ lệch tỷ lệ trên từng tập < 1.0%."""
    df_dedup = pd.read_csv(DEDUP_PATH)
    overall_cat_dist = df_dedup["category_gold"].value_counts(normalize=True)

    cat_map = dict(zip(df_dedup["article_id"], df_dedup["category_gold"]))

    for split_name in ["train", "validation", "test"]:
        filename = f"{split_name}_ids.csv"
        split_ids = pd.read_csv(SPLIT_DIR / filename)["article_id"]
        split_cats = split_ids.map(cat_map).value_counts(normalize=True)

        for cat in overall_cat_dist.index:
            diff = abs(split_cats.get(cat, 0.0) - overall_cat_dist[cat])
            assert diff < 0.01, f"Chuyên mục {cat} trong {split_name} bị lệch {diff:.4f} > 1.0%"


def test_zero_event_leakage_between_train_and_test():
    """
    Kiểm tra yêu cầu 1: Chống rò rỉ cụm sự kiện.
    Các bài viết có độ tương đồng nội dung cao (>= 0.60) hoặc tiêu đề (>= 0.60)
    thuộc cùng một sự kiện phải cùng nằm hoàn toàn trong Train, hoặc Val, hoặc Test.
    """
    df_dedup = pd.read_csv(DEDUP_PATH)
    valid_ids = set(df_dedup["article_id"])

    # Xây dựng lại nhóm sự kiện chuẩn
    group_map = build_event_clusters(
        candidates_path=CANDIDATES_PATH,
        valid_ids=valid_ids,
        content_thresh=0.60,
        title_thresh=0.60,
    )

    train_ids = set(pd.read_csv(SPLIT_DIR / "train_ids.csv")["article_id"])
    val_ids = set(pd.read_csv(SPLIT_DIR / "validation_ids.csv")["article_id"])
    test_ids = set(pd.read_csv(SPLIT_DIR / "test_ids.csv")["article_id"])

    train_groups = set(group_map[aid] for aid in train_ids)
    val_groups = set(group_map[aid] for aid in val_ids)
    test_groups = set(group_map[aid] for aid in test_ids)

    assert len(train_groups & val_groups) == 0, "Có nhóm sự kiện bị cắt đôi giữa Train và Val!"
    assert len(train_groups & test_groups) == 0, "Có nhóm sự kiện bị cắt đôi giữa Train và Test!"
    assert len(val_groups & test_groups) == 0, "Có nhóm sự kiện bị cắt đôi giữa Val và Test!"

