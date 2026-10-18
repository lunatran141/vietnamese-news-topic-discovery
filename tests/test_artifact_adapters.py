"""Kiểm thử adapter chuẩn hóa BERTopic."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.artifacts.adapters import (
    export_bertopic_artifacts,
)


def _create_bertopic_results(
    source_dir: Path,
) -> None:
    """Tạo dữ liệu BERTopic giả phục vụ kiểm thử."""
    source_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = {
        "run_id": "e1_test",
        "embedding_model": "vietnamese-sbert",
        "train_count": 10,
        "validation_count": 5,
        "topic_count": 1,
        "train_outlier_rate": 0.1,
        "validation_outlier_rate": 0.2,
        "topic_diversity": 0.8,
        "runtime_seconds": 12.5,
    }

    (source_dir / "metrics.json").write_text(
        json.dumps(metrics),
        encoding="utf-8",
    )

    with (
        source_dir / "config_used.yaml"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            {
                "embedding_model": "vietnamese-sbert",
                "random_state": 42,
            },
            file,
        )

    pd.DataFrame(
        [
            {
                "topic": 0,
                "topic_name": "0_kinh_doanh",
                "document_count": 9,
                "percentage": 0.9,
                "top_keywords": "kinh doanh, đầu tư",
            }
        ]
    ).to_csv(
        source_dir / "topic_summary.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "article_id": "TTR_001",
                "split": "train",
                "topic": 0,
                "topic_probability": 0.9,
                "is_outlier": False,
                "title": "Tin kinh doanh",
            }
        ]
    ).to_csv(
        source_dir / "document_topics.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "topic": 0,
                "rank": 1,
                "word": "kinh doanh",
                "ctfidf_score": 0.25,
            }
        ]
    ).to_csv(
        source_dir / "topic_keywords.csv",
        index=False,
    )


def test_export_bertopic_artifacts(
    tmp_path: Path,
) -> None:
    """Adapter tạo bộ artifact chuẩn."""
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"

    _create_bertopic_results(source_dir)

    bundle = export_bertopic_artifacts(
        source_dir,
        output_dir,
    )

    assert bundle.model_summary.run_id == "e1_test"
    assert bundle.model_summary.algorithm == "BERTopic"

    assert "topic_id" in bundle.topic_summary.columns
    assert "topic_id" in bundle.document_topics.columns
    assert "keyword" in bundle.topic_keywords.columns
    assert "weight" in bundle.topic_keywords.columns

    assert (
        bundle.topic_keywords.iloc[0]["keyword"]
        == "kinh doanh"
    )


def test_export_bertopic_rejects_missing_files(
    tmp_path: Path,
) -> None:
    """Adapter báo lỗi rõ ràng khi thiếu đầu vào."""
    with pytest.raises(
        FileNotFoundError,
        match="Thiếu file BERTopic",
    ):
        export_bertopic_artifacts(
            tmp_path / "missing",
            tmp_path / "output",
        )