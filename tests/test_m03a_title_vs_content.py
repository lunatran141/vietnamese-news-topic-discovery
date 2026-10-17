"""Kiểm thử tự động cho Task M03A: So sánh chỉ tiêu đề vs Tiêu đề cộng nội dung."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import yaml

from src.data.dataset_loader import load_split_dataset


ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "configs/m03a_title_vs_content.yaml"
RESULTS_DIR = ROOT_DIR / "results/kmeans"


def test_m03a_config_integrity():
    """Kiểm tra tính toàn vẹn và cô lập biến của file cấu hình M03A."""
    assert CONFIG_PATH.exists(), f"Không tìm thấy cấu hình {CONFIG_PATH}"

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert "dataset" in config
    assert "vectorizer" in config
    assert "clustering" in config
    assert "output" in config

    # Đảm bảo giữ nguyên các chốt chặn kỹ thuật
    vec_cfg = config["vectorizer"]
    assert vec_cfg.get("lowercase") is False, "Phải giữ lowercase: false để bảo toàn P3"
    assert vec_cfg.get("stopwords_file") is None, "Phải giữ stopwords_file: null"

    # Kiểm tra khai báo 2 biến thể đầu vào
    variants = config["dataset"]["input_variants"]
    assert len(variants) == 2
    var_ids = [v["id"] for v in variants]
    assert "title_only" in var_ids
    assert "title_content" in var_ids

    # Kiểm tra K khảo sát
    clustering_cfg = config["clustering"]
    assert clustering_cfg.get("primary_k") == 9
    assert 7 in clustering_cfg.get("sensitivity_k_list", [])
    assert 9 in clustering_cfg.get("sensitivity_k_list", [])


def test_m03a_dataset_alignment_and_no_empty_text():
    """Kiểm tra dữ liệu đầu vào: trùng khớp ID 100%, không bài nào bị rỗng."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_path = ROOT_DIR / config["dataset"]["processed_dataset_path"]
    manifest_path = ROOT_DIR / config["dataset"]["split_manifest_path"]

    train_df = load_split_dataset(dataset_path, manifest_path, split_name="train")
    val_df = load_split_dataset(dataset_path, manifest_path, split_name="val")

    assert len(train_df) == 6493, "Tập train phải đủ 6,493 bài"
    assert len(val_df) == 1391, "Tập validation phải đủ 1,391 bài"

    for col in ["processed_title", "text"]:
        assert col in train_df.columns, f"Thiếu cột {col} ở train"
        assert col in val_df.columns, f"Thiếu cột {col} ở validation"

        # Đảm bảo không có bài nào bị rỗng
        train_empty = (train_df[col].fillna("").astype(str).str.strip() == "").sum()
        val_empty = (val_df[col].fillna("").astype(str).str.strip() == "").sum()
        assert train_empty == 0, f"Có {train_empty} bài rỗng ở cột {col} tập train"
        assert val_empty == 0, f"Có {val_empty} bài rỗng ở cột {col} tập validation"


def test_m03a_title_content_matches_m03_baseline():
    """Kiểm tra tính nhất quán: biến thể title_content ở K=9 phải khớp kết quả P3 của M03."""
    metrics_path = RESULTS_DIR / "m03a_title_vs_content_metrics.csv"
    if not metrics_path.exists():
        pytest.skip("Chưa chạy thực nghiệm m03a")

    import pandas as pd
    df = pd.read_csv(metrics_path)

    tc_row = df[df["variant_id"] == "title_content"]
    assert len(tc_row) == 1, "Phải có kết quả của biến thể title_content"
    tc = tc_row.iloc[0]

    # Kiểm tra khớp với baseline P3 của Task M03
    assert abs(tc["purity"] - 0.8728) < 0.015, f"Purity lệch: {tc['purity']}"
    assert abs(tc["nmi"] - 0.7004) < 0.015, f"NMI lệch: {tc['nmi']}"
    assert abs(tc["ari"] - 0.5925) < 0.02, f"ARI lệch: {tc['ari']}"
    assert tc["stopword_keyword_ratio_pct"] == 0.0, "P3 phải sạch 100% từ dừng"


def test_m03a_output_artifacts_integrity():
    """Kiểm tra sự tồn tại và tính đầy đủ của tất cả các file kết quả M03A."""
    metrics_path = RESULTS_DIR / "m03a_title_vs_content_metrics.csv"
    sens_path = RESULTS_DIR / "m03a_title_vs_content_k7_sensitivity.csv"
    breakdown_path = RESULTS_DIR / "m03a_category_breakdown.csv"
    qual_path = RESULTS_DIR / "m03a_qualitative_cluster_examples.csv"
    kws_path = RESULTS_DIR / "m03a_cluster_keywords.json"

    for p in [metrics_path, sens_path, breakdown_path, qual_path, kws_path]:
        assert p.exists(), f"Thiếu file kết quả: {p}"

    import pandas as pd
    metrics_df = pd.read_csv(metrics_path)
    assert len(metrics_df) == 2, "Bảng metrics chính phải có 2 dòng (title_only vs title_content)"
    assert set(metrics_df["variant_id"]) == {"title_only", "title_content"}

    sens_df = pd.read_csv(sens_path)
    assert len(sens_df) == 4, "Bảng độ nhạy phải có 4 dòng (2 biến thể x 2 mức K)"

    breakdown_df = pd.read_csv(breakdown_path)
    assert len(breakdown_df) == 14, "Bảng phân tích chuyên mục phải có 14 dòng (7 chuyên mục x 2 biến thể)"

    qual_df = pd.read_csv(qual_path)
    assert len(qual_df) == 18, "Bảng dẫn chứng định tính phải có 18 dòng (9 cụm x 2 biến thể)"

    with open(kws_path, "r", encoding="utf-8") as f:
        kws_data = json.load(f)
    assert "title_only_k9" in kws_data
    assert "title_content_k9" in kws_data

