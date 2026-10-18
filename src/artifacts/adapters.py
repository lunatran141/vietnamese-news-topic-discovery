"""Chuyển artifact của từng mô hình sang schema chung."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.artifacts.loader import (
    ArtifactBundle,
    load_artifact_bundle,
)


def _read_json(path: Path) -> dict[str, Any]:
    """Đọc file JSON."""
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def _write_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    """Ghi JSON có định dạng dễ đọc."""
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def _require_files(paths: list[Path]) -> None:
    """Kiểm tra các file đầu vào bắt buộc."""
    missing_files = [
        path
        for path in paths
        if not path.is_file()
    ]

    if missing_files:
        missing_text = ", ".join(
            str(path)
            for path in missing_files
        )
        raise FileNotFoundError(
            "Thiếu file BERTopic đầu vào: "
            f"{missing_text}"
        )


def export_bertopic_artifacts(
    source_dir: str | Path,
    output_dir: str | Path,
) -> ArtifactBundle:
    """Chuyển kết quả BERTopic sang artifact schema chung."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    metrics_path = source_dir / "metrics.json"
    config_path = source_dir / "config_used.yaml"
    topic_summary_path = source_dir / "topic_summary.csv"
    document_topics_path = source_dir / "document_topics.csv"
    topic_keywords_path = source_dir / "topic_keywords.csv"

    _require_files(
        [
            metrics_path,
            config_path,
            topic_summary_path,
            document_topics_path,
            topic_keywords_path,
        ]
    )

    metrics = _read_json(metrics_path)

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file) or {}

    topic_summary = pd.read_csv(
        topic_summary_path,
        encoding="utf-8-sig",
    ).rename(
        columns={
            "topic": "topic_id",
        }
    )

    document_topics = pd.read_csv(
        document_topics_path,
        encoding="utf-8-sig",
    ).rename(
        columns={
            "topic": "topic_id",
        }
    )

    topic_keywords = pd.read_csv(
        topic_keywords_path,
        encoding="utf-8-sig",
    ).rename(
        columns={
            "topic": "topic_id",
            "word": "keyword",
            "ctfidf_score": "weight",
        }
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalized_topic_summary_path = (
        output_dir / "topic_summary.csv"
    )
    normalized_document_topics_path = (
        output_dir / "document_topics.csv"
    )
    normalized_topic_keywords_path = (
        output_dir / "topic_keywords.csv"
    )
    normalized_config_path = (
        output_dir / "config_used.yaml"
    )

    topic_summary.to_csv(
        normalized_topic_summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    document_topics.to_csv(
        normalized_document_topics_path,
        index=False,
        encoding="utf-8-sig",
    )
    topic_keywords.to_csv(
        normalized_topic_keywords_path,
        index=False,
        encoding="utf-8-sig",
    )

    shutil.copy2(
        config_path,
        normalized_config_path,
    )

    run_id = str(
        metrics.get(
            "run_id",
            source_dir.name,
        )
    )
    embedding_model = str(
        metrics.get(
            "embedding_model",
            config.get(
                "embedding_model",
                "unknown",
            ),
        )
    )

    model_summary = {
        "model_name": f"BERTopic {run_id}",
        "run_id": run_id,
        "algorithm": "BERTopic",
        "representation": embedding_model,
        "topic_count": int(
            metrics.get(
                "topic_count",
                len(
                    topic_summary[
                        topic_summary["topic_id"] != -1
                    ]
                ),
            )
        ),
        "outlier_rate": metrics.get(
            "validation_outlier_rate"
        ),
        "runtime_seconds": metrics.get(
            "runtime_seconds"
        ),
        "metrics": {
            "topic_diversity": metrics.get(
                "topic_diversity"
            ),
            "train_outlier_rate": metrics.get(
                "train_outlier_rate"
            ),
            "validation_outlier_rate": metrics.get(
                "validation_outlier_rate"
            ),
            "train_count": metrics.get(
                "train_count"
            ),
            "validation_count": metrics.get(
                "validation_count"
            ),
        },
        "config": config,
    }

    manifest = {
        "schema_version": "1.0",
        "model_name": model_summary["model_name"],
        "run_id": run_id,
        "model_summary_path": "model_summary.json",
        "topic_summary_path": "topic_summary.csv",
        "document_topics_path": "document_topics.csv",
        "topic_keywords_path": "topic_keywords.csv",
        "config_path": "config_used.yaml",
    }

    _write_json(
        output_dir / "model_summary.json",
        model_summary,
    )
    _write_json(
        output_dir / "manifest.json",
        manifest,
    )

    return load_artifact_bundle(output_dir)