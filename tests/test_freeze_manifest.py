"""
Kiểm thử tự động cho Task D04: Review & Freeze Dataset v1.
Module: tests/test_freeze_manifest.py

Xác thực:
- File dataset_manifest.json tồn tại, hợp lệ theo schema và được phê duyệt (review_status == approved).
- Các mã băm SHA-256 của Raw, Configs và 4 biến thể P0–P3 khớp tuyệt đối giữa manifest và đĩa vật lý.
- 4 biến thể P0–P3 bảo toàn đúng 9,274 dòng, không chứa rỗng/null, và chia sẻ cùng danh sách article_id.
- Toàn bộ pipeline tái lập và cấu hình không bị trôi .
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pytest

from src.preprocessing.freeze_dataset import compute_sha256, verify_manifest, EXPECTED_SCHEMA


MANIFEST_PATH = Path("data/processed/dataset_manifest.json")
CONFIG_MANIFEST_PATH = Path("configs/dataset_manifest.json")


def test_manifest_files_exist():
    """Kiểm tra sự tồn tại của file manifest tại data/processed/ và configs/."""
    assert MANIFEST_PATH.exists(), f"Không tìm thấy manifest tại: {MANIFEST_PATH}"
    assert CONFIG_MANIFEST_PATH.exists(), f"Không tìm thấy manifest bản sao tại: {CONFIG_MANIFEST_PATH}"


def test_manifest_content_schema():
    """Kiểm tra cấu trúc và metadata bắt buộc trong dataset_manifest.json."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest.get("dataset_version") == "v1.0"
    assert manifest.get("review_status") == "approved"
    assert manifest.get("quality_gate") == "PASSED"
    assert manifest.get("random_seed") == 42

    assert "raw_data" in manifest
    assert "dedup_data" in manifest
    assert "configs" in manifest
    assert "schema" in manifest
    assert "variants" in manifest
    assert "model_recommendations" in manifest
    assert "reproduction_commands" in manifest

    assert manifest["schema"]["columns"] == EXPECTED_SCHEMA
    assert manifest["schema"]["model_input_column"] == "text"
    assert manifest["schema"]["formula_for_text"] == (
        "processed_title + ' ' + processed_content"
    )


def test_manifest_verification_function():
    """Kiểm tra hàm verify_manifest trả về True với dữ liệu hiện tại."""
    assert verify_manifest(MANIFEST_PATH) is True


def test_raw_and_dedup_row_counts_and_hashes():
    """Kiểm tra số dòng và hash của raw data và interim deduplicated data."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    raw_meta = manifest["raw_data"]
    raw_path = Path(raw_meta["file_path"])
    assert raw_path.exists()
    assert compute_sha256(raw_path) == raw_meta["sha256"]

    dedup_meta = manifest["dedup_data"]
    dedup_path = Path(dedup_meta["file_path"])
    assert dedup_path.exists()
    assert compute_sha256(dedup_path) == dedup_meta["sha256"]
    assert dedup_meta["row_count"] == 9274
    assert dedup_meta["dropped_duplicates_count"] == 30


def test_processed_variants_integrity_and_alignment():
    """Kiểm tra 4 biến thể P0–P3 đồng bộ 100% về số dòng, thứ tự article_id và không có text rỗng."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    dedup_df = pd.read_csv(manifest["dedup_data"]["file_path"])
    expected_ids = dedup_df["article_id"].tolist()

    variants = manifest["variants"]
    expected_variants = ["p0_minimal", "p1_normalized", "p2_segmented", "p3_no_stopwords"]

    for var_key in expected_variants:
        assert var_key in variants, f"Thiếu biến thể {var_key} trong manifest"
        var_meta = variants[var_key]
        var_path = Path(var_meta["file_path"])
        assert var_path.exists(), f"Không tìm thấy file: {var_path}"

        # Kiểm tra hash
        assert compute_sha256(var_path) == var_meta["sha256"]

        # Đọc dữ liệu và kiểm tra cấu trúc
        df = pd.read_csv(var_path)
        assert len(df) == 9274
        assert list(df.columns) == EXPECTED_SCHEMA
        assert df["article_id"].tolist() == expected_ids
        assert df["text"].isna().sum() == 0
        assert (df["text"].str.strip() == "").sum() == 0


def test_configs_and_tokenizers():
    """Kiểm tra các file cấu hình và thông tin tokenizer trong manifest."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    configs_meta = manifest["configs"]
    prep_cfg = Path(configs_meta["preprocessing_yaml"]["file_path"])
    stopwords_cfg = Path(configs_meta["vietnamese_stopwords"]["file_path"])
    split_cfg = Path(configs_meta["split_yaml"]["file_path"])

    assert prep_cfg.exists()
    assert compute_sha256(prep_cfg) == configs_meta["preprocessing_yaml"]["sha256"]

    assert stopwords_cfg.exists()
    assert compute_sha256(stopwords_cfg) == configs_meta["vietnamese_stopwords"]["sha256"]

    assert split_cfg.exists()
    assert compute_sha256(split_cfg) == configs_meta["split_yaml"]["sha256"]

    assert configs_meta["tokenizer"]["name"] == "pyvi"
    assert configs_meta["tokenizer"]["version"] == "0.1.1"

