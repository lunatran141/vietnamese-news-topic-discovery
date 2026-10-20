"""Kiểm thử đọc Parquet NMF và xuất common artifacts."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.artifacts.adapters import export_nmf_artifacts
from src.artifacts.nmf_parquet import (
    normalize_nmf_topic_id,
    read_nmf_parquet_assignments,
)


def _write_nmf_source(source_dir: Path, result_dir: Path) -> None:
    """Tạo một run NMF nhỏ để kiểm thử exporter không phụ thuộc dữ liệu thật."""
    source_dir.mkdir(parents=True)
    result_dir.mkdir(parents=True)
    development = pd.DataFrame(
        [
            {
                "article_id": "A1",
                "split": "train",
                "topic_00_share": 0.8,
                "topic_01_share": 0.2,
            },
            {
                "article_id": "A2",
                "split": "val",
                "topic_00_share": 0.1,
                "topic_01_share": 0.9,
            },
        ]
    )
    test = pd.DataFrame(
        [
            {
                "article_id": "A3",
                "split": "test",
                "title": "Article three",
                "source": "VNE",
                "published_at": "2026-10-01",
                "primary_topic_id": "NMF_00",
                "primary_topic_share": 0.7,
                "secondary_topic_id": "NMF_01",
                "secondary_topic_share": 0.3,
                "assigned_topic_ids": '["NMF_00", "NMF_01"]',
                "is_multi_topic": True,
                "topic_00_share": 0.7,
                "topic_01_share": 0.3,
            },
            {
                "article_id": "A4",
                "split": "test",
                "title": "Article four",
                "source": "TTR",
                "published_at": "2026-10-02",
                "primary_topic_id": "NMF_01",
                "primary_topic_share": 0.8,
                "secondary_topic_id": "NMF_00",
                "secondary_topic_share": 0.8,
                "assigned_topic_ids": '["NMF_01", "NMF_00"]',
                "is_multi_topic": True,
                "topic_00_share": 0.2,
                "topic_01_share": 0.8,
            },
        ]
    )
    development.to_parquet(
        source_dir / "development_document_topics.parquet", index=False
    )
    test.to_parquet(source_dir / "document_topics.parquet", index=False)
    (source_dir / "config.json").write_text(
        json.dumps({"model": {"random_seed": 42}}), encoding="utf-8"
    )
    (source_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_id": source_dir.name,
                "selected_n_topics": 2,
                "split_counts": {"train": 1, "val": 1, "test": 2},
            }
        ),
        encoding="utf-8",
    )
    (source_dir / "topic_keywords.json").write_text(
        json.dumps(
            [
                {
                    "topic_id": "NMF_00",
                    "keywords": [{"term": "business", "weight": 0.8}],
                },
                {
                    "topic_id": "NMF_01",
                    "keywords": [{"term": "sports", "weight": 0.6}],
                },
            ]
        ),
        encoding="utf-8",
    )
    (source_dir / "representative_docs.json").write_text(
        json.dumps(
            [
                {
                    "topic_id": "NMF_00",
                    "rank": 1,
                    "article_id": "A3",
                    "title": "Article three",
                    "topic_share": 0.7,
                },
                {
                    "topic_id": "NMF_01",
                    "rank": 1,
                    "article_id": "A4",
                    "title": "Article four",
                    "topic_share": 0.8,
                },
            ]
        ),
        encoding="utf-8",
    )
    (source_dir / "topic_display_mapping.json").write_text(
        json.dumps(
            {
                "topics": [
                    {"topic_id": "NMF_00", "display_label": "Business"},
                    {"topic_id": "NMF_01", "display_label": "Sports"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (result_dir / "test_metrics.json").write_text(
        json.dumps(
            {
                "npmi_coherence": 0.4,
                "topic_diversity": 1.0,
                "converged": True,
            }
        ),
        encoding="utf-8",
    )


def test_normalize_nmf_topic_id() -> None:
    """ID gốc NMF thành số nguyên, outlier giữ -1."""
    assert normalize_nmf_topic_id("NMF_03") == 3
    assert normalize_nmf_topic_id(2) == 2
    assert normalize_nmf_topic_id("-1") == -1
    with pytest.raises(ValueError, match="không hợp lệ"):
        normalize_nmf_topic_id("NMF_invalid")


def test_read_nmf_parquet_assignments_normalizes_topic_ids(
    tmp_path: Path,
) -> None:
    """Adapter ghép split và giữ trọng số NMF ngoài trường xác suất."""
    source_dir = tmp_path / "nmf_run"
    result_dir = tmp_path / "results"
    _write_nmf_source(source_dir, result_dir)

    assignments = read_nmf_parquet_assignments(source_dir)

    assert set(assignments["topic_id"]) == {0, 1}
    assert set(assignments["split"]) == {"train", "val", "test"}
    assert assignments["topic_probability"].isna().all()
    assert assignments["is_outlier"].eq(False).all()
    secondary_topic_id = assignments.loc[
        assignments.article_id == "A3",
        "secondary_topic_id",
    ].item()
    assert secondary_topic_id == 1


def test_export_nmf_artifacts_loads_common_bundle(
    tmp_path: Path,
) -> None:
    """Exporter tạo bundle có topic ID nguyên và xác suất để trống."""
    source_dir = tmp_path / "nmf_p3_test_run"
    result_dir = tmp_path / "results"
    output_dir = tmp_path / "common"
    _write_nmf_source(source_dir, result_dir)

    bundle = export_nmf_artifacts(source_dir, output_dir, result_dir)

    assert bundle.model_summary.algorithm == "NMF"
    assert bundle.model_summary.topic_count == 2
    assert set(bundle.document_topics["topic_id"]) == {0, 1}
    assert bundle.document_topics["topic_probability"].isna().all()
    assert bundle.topic_summary["document_count"].sum() == 4
    assert set(bundle.topic_keywords["topic_id"]) == {0, 1}
    assert (output_dir / "representative_docs.csv").is_file()
    assert (output_dir / "projection_2d.csv").is_file()


def test_read_nmf_parquet_rejects_mismatched_primary_topic(
    tmp_path: Path,
) -> None:
    """ID chủ đề ghi trong Parquet phải khớp topic share lớn nhất."""
    source_dir = tmp_path / "nmf_run"
    result_dir = tmp_path / "results"
    _write_nmf_source(source_dir, result_dir)
    test_path = source_dir / "document_topics.parquet"
    test = pd.read_parquet(test_path)
    test.loc[0, "primary_topic_id"] = "NMF_01"
    test.to_parquet(test_path, index=False)

    with pytest.raises(ValueError, match="không khớp topic shares"):
        read_nmf_parquet_assignments(source_dir)
