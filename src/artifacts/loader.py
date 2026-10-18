"""Đọc bộ artifact chuẩn để phục vụ API và Streamlit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.artifacts.schemas import (
    ArtifactManifest,
    ModelSummary,
)


TOPIC_SUMMARY_COLUMNS = {
    "topic_id",
    "topic_name",
    "document_count",
    "percentage",
    "top_keywords",
}

DOCUMENT_TOPIC_COLUMNS = {
    "article_id",
    "split",
    "topic_id",
    "topic_probability",
    "is_outlier",
}

TOPIC_KEYWORD_COLUMNS = {
    "topic_id",
    "rank",
    "keyword",
    "weight",
}


@dataclass(frozen=True)
class ArtifactBundle:
    """Toàn bộ dữ liệu cần cho API và giao diện."""

    manifest: ArtifactManifest
    model_summary: ModelSummary
    topic_summary: pd.DataFrame
    document_topics: pd.DataFrame
    topic_keywords: pd.DataFrame


def _load_json(path: Path) -> dict:
    """Đọc một file JSON."""
    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"File JSON không hợp lệ: {path}"
        ) from error


def _resolve_artifact_path(
    artifact_dir: Path,
    relative_path: str,
) -> Path:
    """Tạo đường dẫn và không cho thoát khỏi artifact directory."""
    root = artifact_dir.resolve()
    resolved = (root / relative_path).resolve()

    if root not in resolved.parents and resolved != root:
        raise ValueError(
            "Đường dẫn artifact nằm ngoài thư mục chạy: "
            f"{relative_path}"
        )

    return resolved


def _require_file(path: Path) -> None:
    """Báo lỗi rõ ràng nếu artifact bị thiếu."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy artifact bắt buộc: {path}"
        )


def _validate_columns(
    frame: pd.DataFrame,
    required_columns: set[str],
    artifact_name: str,
) -> None:
    """Kiểm tra các cột bắt buộc của một bảng."""
    missing_columns = required_columns - set(frame.columns)

    if missing_columns:
        missing_text = ", ".join(
            sorted(missing_columns)
        )
        raise ValueError(
            f"{artifact_name} thiếu cột: {missing_text}"
        )


def load_artifact_bundle(
    artifact_dir: str | Path,
) -> ArtifactBundle:
    """Đọc và kiểm tra một bộ artifact chuẩn."""
    artifact_dir = Path(artifact_dir)

    manifest_path = artifact_dir / "manifest.json"
    _require_file(manifest_path)

    manifest = ArtifactManifest.model_validate(
        _load_json(manifest_path)
    )

    model_summary_path = _resolve_artifact_path(
        artifact_dir,
        manifest.model_summary_path,
    )
    topic_summary_path = _resolve_artifact_path(
        artifact_dir,
        manifest.topic_summary_path,
    )
    document_topics_path = _resolve_artifact_path(
        artifact_dir,
        manifest.document_topics_path,
    )
    topic_keywords_path = _resolve_artifact_path(
        artifact_dir,
        manifest.topic_keywords_path,
    )
    config_path = _resolve_artifact_path(
        artifact_dir,
        manifest.config_path,
    )

    required_paths = [
        model_summary_path,
        topic_summary_path,
        document_topics_path,
        topic_keywords_path,
        config_path,
    ]

    for path in required_paths:
        _require_file(path)

    model_summary = ModelSummary.model_validate(
        _load_json(model_summary_path)
    )

    topic_summary = pd.read_csv(
        topic_summary_path,
        encoding="utf-8-sig",
    )
    document_topics = pd.read_csv(
        document_topics_path,
        encoding="utf-8-sig",
    )
    topic_keywords = pd.read_csv(
        topic_keywords_path,
        encoding="utf-8-sig",
    )

    _validate_columns(
        topic_summary,
        TOPIC_SUMMARY_COLUMNS,
        "topic_summary.csv",
    )
    _validate_columns(
        document_topics,
        DOCUMENT_TOPIC_COLUMNS,
        "document_topics.csv",
    )
    _validate_columns(
        topic_keywords,
        TOPIC_KEYWORD_COLUMNS,
        "topic_keywords.csv",
    )

    if manifest.model_name != model_summary.model_name:
        raise ValueError(
            "model_name trong manifest và model summary "
            "không giống nhau"
        )

    if manifest.run_id != model_summary.run_id:
        raise ValueError(
            "run_id trong manifest và model summary "
            "không giống nhau"
        )

    return ArtifactBundle(
        manifest=manifest,
        model_summary=model_summary,
        topic_summary=topic_summary,
        document_topics=document_topics,
        topic_keywords=topic_keywords,
    )