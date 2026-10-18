"""Kiểm thử loader dành cho API và Streamlit."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.artifacts.loader import load_artifact_bundle


def _write_json(
    path: Path,
    data: dict,
) -> None:
    """Ghi JSON phục vụ kiểm thử."""
    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _create_valid_artifacts(
    artifact_dir: Path,
) -> None:
    """Tạo một bộ artifact chuẩn tối thiểu."""
    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    _write_json(
        artifact_dir / "manifest.json",
        {
            "schema_version": "1.0",
            "model_name": "BERTopic E1",
            "run_id": "e1_test",
            "model_summary_path": "model_summary.json",
            "topic_summary_path": "topic_summary.csv",
            "document_topics_path": "document_topics.csv",
            "topic_keywords_path": "topic_keywords.csv",
            "config_path": "config_used.json",
        },
    )

    _write_json(
        artifact_dir / "model_summary.json",
        {
            "model_name": "BERTopic E1",
            "run_id": "e1_test",
            "algorithm": "BERTopic",
            "representation": "Vietnamese SBERT",
            "topic_count": 1,
            "outlier_rate": 0.1,
            "runtime_seconds": 12.5,
            "metrics": {
                "topic_diversity": 0.75,
            },
            "config": {},
        },
    )

    _write_json(
        artifact_dir / "config_used.json",
        {
            "random_state": 42,
        },
    )

    pd.DataFrame(
        [
            {
                "topic_id": 0,
                "topic_name": "Kinh doanh",
                "document_count": 9,
                "percentage": 0.9,
                "top_keywords": "doanh nghiệp, đầu tư",
            }
        ]
    ).to_csv(
        artifact_dir / "topic_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [
            {
                "article_id": "TTR_001",
                "split": "train",
                "topic_id": 0,
                "topic_probability": 0.91,
                "is_outlier": False,
                "title": "Tin kinh doanh",
            }
        ]
    ).to_csv(
        artifact_dir / "document_topics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [
            {
                "topic_id": 0,
                "rank": 1,
                "keyword": "doanh nghiệp",
                "weight": 0.2,
            }
        ]
    ).to_csv(
        artifact_dir / "topic_keywords.csv",
        index=False,
        encoding="utf-8-sig",
    )


def test_load_artifact_bundle(
    tmp_path: Path,
) -> None:
    """Loader đọc được bộ artifact hợp lệ."""
    artifact_dir = tmp_path / "bertopic"
    _create_valid_artifacts(artifact_dir)

    bundle = load_artifact_bundle(artifact_dir)

    assert bundle.manifest.schema_version == "1.0"
    assert bundle.model_summary.algorithm == "BERTopic"
    assert len(bundle.topic_summary) == 1
    assert len(bundle.document_topics) == 1
    assert len(bundle.topic_keywords) == 1


def test_loader_rejects_missing_manifest(
    tmp_path: Path,
) -> None:
    """Loader báo lỗi nếu thiếu manifest."""
    with pytest.raises(
        FileNotFoundError,
        match="manifest.json",
    ):
        load_artifact_bundle(tmp_path)


def test_loader_rejects_missing_columns(
    tmp_path: Path,
) -> None:
    """Loader báo lỗi nếu bảng thiếu cột bắt buộc."""
    artifact_dir = tmp_path / "bertopic"
    _create_valid_artifacts(artifact_dir)

    pd.DataFrame(
        [
            {
                "topic_id": 0,
                "topic_name": "Kinh doanh",
            }
        ]
    ).to_csv(
        artifact_dir / "topic_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with pytest.raises(
        ValueError,
        match="document_count",
    ):
        load_artifact_bundle(artifact_dir)


def test_loader_rejects_mismatched_run_id(
    tmp_path: Path,
) -> None:
    """Manifest và model summary phải cùng run_id."""
    artifact_dir = tmp_path / "bertopic"
    _create_valid_artifacts(artifact_dir)

    summary_path = artifact_dir / "model_summary.json"
    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )
    summary["run_id"] = "different_run"
    _write_json(summary_path, summary)

    with pytest.raises(
        ValueError,
        match="run_id",
    ):
        load_artifact_bundle(artifact_dir)