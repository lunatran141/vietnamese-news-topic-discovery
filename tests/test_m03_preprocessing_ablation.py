"""Kiểm thử tự động cho Task M03: Preprocessing Ablation P0 -> P3"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.kmeans.run_m03_preprocessing_ablation import (
    compute_lexical_interpretability_metrics,
)
from src.features.tfidf import create_tfidf_vectorizer


@pytest.fixture
def m03_config() -> dict[str, Any]:
    cfg_path = Path("configs/m03_preprocessing_ablation.yaml")
    assert cfg_path.exists(), f"Không tìm thấy file cấu hình {cfg_path}"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_m03_config_isolation(m03_config: dict[str, Any]) -> None:
    """Đảm bảo cấu hình TF-IDF trong M03 duy trì tính cô lập biến thể"""
    vec_cfg = m03_config["vectorizer"]
    assert vec_cfg.get("lowercase") is False, "TF-IDF vectorizer phải có lowercase=False để bảo toàn phân biệt P0 vs P1"
    assert vec_cfg.get("stopwords_file") is None, "TF-IDF vectorizer phải có stopwords_file=None để bảo toàn phân biệt P2 vs P3"

    variants = m03_config["dataset"]["variants"]
    assert len(variants) == 4
    variant_ids = [v["id"] for v in variants]
    assert variant_ids == ["P0", "P1", "P2", "P3"]

    clustering_cfg = m03_config["clustering"]
    assert clustering_cfg["primary_k"] == 9
    assert 7 in clustering_cfg["sensitivity_k_list"]
    assert 9 in clustering_cfg["sensitivity_k_list"]


def test_tfidf_lowercase_parameter_preservation() -> None:
    """Kiểm tra TfidfVectorizer bảo toàn chữ in hoa khi lowercase=False."""
    docs = ["Hà_Nội là thủ_đô", "hà_nội là thủ_đô"]
    
    # Vectorizer với lowercase=False
    vec_no_lower = create_tfidf_vectorizer({
        "lowercase": False,
        "min_df": 1,
        "max_df": 1.0,
        "token_pattern": r"(?u)\b\w+\b",
    })
    vec_no_lower.fit(docs)
    features_no_lower = set(vec_no_lower.get_feature_names_out())
    assert "Hà_Nội" in features_no_lower
    assert "hà_nội" in features_no_lower

    # Vectorizer với lowercase=True
    vec_lower = create_tfidf_vectorizer({
        "lowercase": True,
        "min_df": 1,
        "max_df": 1.0,
        "token_pattern": r"(?u)\b\w+\b",
    })
    vec_lower.fit(docs)
    features_lower = set(vec_lower.get_feature_names_out())
    assert "Hà_Nội" not in features_lower
    assert "hà_nội" in features_lower


def test_compute_lexical_interpretability_metrics() -> None:
    """Kiểm tra hàm tính các chỉ số diễn giải từ vựng và tỷ lệ nhiễu từ dừng."""
    cluster_kws = {
        "0": [
            {"term": "bác_sĩ", "weight": 0.5},
            {"term": "điều_trị", "weight": 0.4},
            {"term": "và", "weight": 0.3},
            {"term": "bệnh_viện", "weight": 0.25},
            {"term": "của", "weight": 0.2},
            {"term": "thuốc", "weight": 0.15},
            {"term": "sức_khỏe", "weight": 0.12},
            {"term": "y_tế", "weight": 0.11},
            {"term": "trong", "weight": 0.10},
            {"term": "bệnh_nhân", "weight": 0.09},
        ],
        "1": [
            {"term": "chứng_khoán", "weight": 0.6},
            {"term": "cổ_phiếu", "weight": 0.5},
            {"term": "thị_trường", "weight": 0.4},
            {"term": "là", "weight": 0.3},
            {"term": "đầu_tư", "weight": 0.2},
            {"term": "giá", "weight": 0.18},
            {"term": "tăng", "weight": 0.16},
            {"term": "doanh_nghiệp", "weight": 0.14},
            {"term": "có", "weight": 0.12},
            {"term": "lợi_nhuận", "weight": 0.11},
        ],
    }
    stopwords = {"và", "của", "trong", "là", "có"}

    metrics = compute_lexical_interpretability_metrics(cluster_kws, stopwords)

    assert 0.0 <= metrics["topic_diversity_top10"] <= 1.0

    assert metrics["compound_keyword_ratio_pct"] == 60.0

    assert metrics["stopword_keyword_ratio_pct"] == 25.0
    
    assert metrics["stopword_pollution_count_top15"] == 5


def test_m03_output_files_integrity() -> None:
    """Kiểm tra sự tồn tại và schema của các file kết quả M03 nếu đã sinh ra."""
    metrics_path = Path("results/kmeans/m03_preprocessing_ablation_metrics.csv")
    sens_path = Path("results/kmeans/m03_preprocessing_ablation_k7_sensitivity.csv")
    qual_path = Path("results/kmeans/m03_qualitative_cluster_examples.csv")
    kws_path = Path("results/kmeans/m03_cluster_keywords_p0_p3.json")

    # Nếu các file đã chạy xong thì kiểm tra cấu trúc schema
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        expected_cols = [
            "variant_id",
            "variant_name",
            "k",
            "vocab_size",
            "val_sparsity_pct",
            "val_oov_rate_pct",
            "silhouette_score",
            "davies_bouldin_index",
            "calinski_harabasz_score",
            "ari",
            "nmi",
            "purity",
            "stability_ari_mean",
            "compound_keyword_ratio_pct",
            "stopword_keyword_ratio_pct",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Thiếu cột {col} trong metrics.csv"

        assert set(df["variant_id"]) == {"P0", "P1", "P2", "P3"}
        assert len(df) == 4

        # P3 phải có tỷ lệ nhiễu từ dừng gần 0% hoặc 0%
        p3_row = df[df["variant_id"] == "P3"].iloc[0]
        p0_row = df[df["variant_id"] == "P0"].iloc[0]
        assert p3_row["stopword_keyword_ratio_pct"] < p0_row["stopword_keyword_ratio_pct"]

    if sens_path.exists():
        sens_df = pd.read_csv(sens_path)
        assert set(sens_df["k"]) == {7, 9}
        assert len(sens_df) == 8

    if qual_path.exists():
        qual_df = pd.read_csv(qual_path)
        assert "cluster_id" in qual_df.columns
        assert "top10_keywords" in qual_df.columns
        assert "dominant_gold_category" in qual_df.columns

    if kws_path.exists():
        with open(kws_path, "r", encoding="utf-8") as f:
            kws = json.load(f)
        assert "P0_k9" in kws
        assert "P3_k9" in kws
