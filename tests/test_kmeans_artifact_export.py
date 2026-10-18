"""Kiểm thử bộ chuyển đổi common artifact cho mô hình K-Means."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.artifacts.adapters import (
    export_kmeans_artifacts,
)
from src.artifacts.loader import (
    load_artifact_bundle,
)


REQUIRED_EXPORT_FILES = (
    "manifest.json",
    "model_summary.json",
    "config_used.yaml",
    "topic_summary.csv",
    "document_topics.csv",
    "topic_keywords.csv",
)


def _create_kmeans_source_dir(source_dir: Path) -> None:
    """Tạo bộ dữ liệu K-Means giả lập phục vụ kiểm thử đơn vị."""
    source_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "article_id": "TTR_001",
                "title": "Lãi suất ngân hàng giảm",
                "cluster_id": 0,
                "distance_to_center": 0.91,
                "split": "train",
                "category_gold": "Kinh doanh",
            },
            {
                "article_id": "TTR_002",
                "title": "Thị trường chứng khoán khởi sắc",
                "cluster_id": 0,
                "distance_to_center": 0.94,
                "split": "val",
                "category_gold": "Kinh doanh",
            },
            {
                "article_id": "TTR_003",
                "title": "Đội tuyển bóng đá vào chung kết",
                "cluster_id": 1,
                "distance_to_center": 0.89,
                "split": "train",
                "category_gold": "Thể thao",
            },
        ]
    ).to_csv(
        source_dir / "document_topics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    topic_keywords = {
        "0": [
            {"term": "ngân_hàng", "weight": 0.08},
            {"term": "lãi_suất", "weight": 0.06},
            {"term": "chứng_khoán", "weight": 0.05},
        ],
        "1": [
            {"term": "bóng_đá", "weight": 0.09},
            {"term": "chung_kết", "weight": 0.07},
            {"term": "đội_tuyển", "weight": 0.04},
        ],
    }
    (source_dir / "topic_keywords.json").write_text(
        json.dumps(topic_keywords, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    run_metadata = {
        "pipeline_name": "M02_kmeans_baseline",
        "execution_duration_seconds": 45.2,
        "n_clusters": 2,
        "random_state": 42,
        "metrics_validation": {
            "inertia": 120.5,
            "silhouette_score": 0.025,
            "davies_bouldin_index": 5.1,
            "calinski_harabasz_score": 12.3,
            "ari": 0.61,
            "nmi": 0.72,
            "purity": 0.88,
        },
    }
    (source_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    config = {
        "dataset": {
            "processed_dataset_path": "data/processed/p2_segmented.csv",
            "data_version": "v1.0",
        },
        "clustering": {
            "algorithm": "kmeans",
            "selected_k": 2,
            "random_state": 42,
        },
    }
    (source_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_export_kmeans_artifacts_meets_all_schema_requirements(
    tmp_path: Path,
) -> None:
    """Kiểm tra đủ 5 tiêu chí bắt buộc của exporter K-Means."""
    source_dir = tmp_path / "kmeans_source"
    output_dir = tmp_path / "kmeans_baseline"
    _create_kmeans_source_dir(source_dir)

    input_df = pd.read_csv(
        source_dir / "document_topics.csv",
        encoding="utf-8-sig",
    )

    bundle = export_kmeans_artifacts(
        source_dir=source_dir,
        output_dir=output_dir,
        run_id="kmeans_baseline",
    )

    # 1. Export đủ sáu file
    for filename in REQUIRED_EXPORT_FILES:
        assert (output_dir / filename).is_file(), (
            f"Thiếu file bắt buộc: {filename}"
        )

    # 2. Số bài sau export bằng số bài đầu vào
    assert len(bundle.document_topics) == len(input_df)

    # 3. cluster_id được đổi thành topic_id; term đổi thành keyword
    assert "topic_id" in bundle.document_topics.columns
    assert "cluster_id" not in bundle.document_topics.columns
    assert "keyword" in bundle.topic_keywords.columns
    assert "term" not in bundle.topic_keywords.columns

    # 4. topic_probability được phép rỗng và không dùng 1 - distance_to_center
    assert bundle.document_topics["topic_probability"].isna().all()
    assert (~bundle.document_topics["is_outlier"].astype(bool)).all()
    assert "distance_to_center" in bundle.document_topics.columns
    assert "category_gold" in bundle.document_topics.columns

    # 5. Loader đọc được output
    reloaded = load_artifact_bundle(output_dir)
    assert reloaded.model_summary.algorithm == "KMeans"
    assert reloaded.model_summary.run_id == "kmeans_baseline"
    assert reloaded.model_summary.topic_count == 2
    assert reloaded.model_summary.outlier_rate == 0.0
    assert len(reloaded.topic_summary) == 2


def test_export_kmeans_rejects_missing_files(
    tmp_path: Path,
) -> None:
    """Báo lỗi rõ ràng khi thiếu file đầu vào của K-Means."""
    with pytest.raises(
        FileNotFoundError,
        match="Thiếu file KMeans",
    ):
        export_kmeans_artifacts(
            source_dir=tmp_path / "missing",
            output_dir=tmp_path / "output",
        )


def test_real_kmeans_baseline_bundle_is_loadable() -> None:
    """Kiểm tra trực tiếp bộ artifact thật tại artifacts/common/kmeans/kmeans_baseline."""
    real_source = Path("artifacts/kmeans/document_topics.csv")
    real_output = Path("artifacts/common/kmeans/kmeans_baseline")
    if not real_source.is_file() or not real_output.is_dir():
        pytest.skip("Chưa có dữ liệu artifact thật trên máy")

    for filename in REQUIRED_EXPORT_FILES:
        assert (real_output / filename).is_file()

    input_df = pd.read_csv(real_source, encoding="utf-8-sig")
    bundle = load_artifact_bundle(real_output)

    assert len(bundle.document_topics) == len(input_df)
    assert "topic_id" in bundle.document_topics.columns
    assert "cluster_id" not in bundle.document_topics.columns
    assert bundle.document_topics["topic_probability"].isna().all()
    assert (~bundle.document_topics["is_outlier"].astype(bool)).all()
    assert "distance_to_center" in bundle.document_topics.columns
    assert len(bundle.topic_summary) == 9
    assert len(bundle.topic_keywords) == 135

