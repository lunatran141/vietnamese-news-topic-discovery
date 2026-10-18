"""Cung cấp dữ liệu artifact cho giao diện Streamlit."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.artifacts.loader import (
    ArtifactBundle,
    load_artifact_bundle,
)


DEFAULT_ARTIFACT_DIR = Path(
    os.getenv(
        "TOPIC_ARTIFACT_DIR",
        (
            "artifacts/common/bertopic/"
            "e1_vietnamese_sbert"
        ),
    )
)


@st.cache_resource(show_spinner=False)
def load_dashboard_bundle(
    artifact_dir: str = str(DEFAULT_ARTIFACT_DIR),
) -> ArtifactBundle:
    """Đọc và cache toàn bộ artifact của mô hình."""
    return load_artifact_bundle(artifact_dir)


def get_model_overview(
    bundle: ArtifactBundle,
) -> dict[str, int | float | str | None]:
    """Tạo các số liệu tổng quan cho dashboard."""
    summary = bundle.model_summary
    documents = bundle.document_topics

    assigned_documents = int(
        (~documents["is_outlier"].astype(bool)).sum()
    )
    outlier_documents = int(
        documents["is_outlier"].astype(bool).sum()
    )

    return {
        "model_name": summary.model_name,
        "algorithm": summary.algorithm,
        "representation": summary.representation,
        "topic_count": summary.topic_count,
        "document_count": len(documents),
        "assigned_document_count": assigned_documents,
        "outlier_document_count": outlier_documents,
        "outlier_rate": summary.outlier_rate,
        "runtime_seconds": summary.runtime_seconds,
        "topic_diversity": summary.metrics.get(
            "topic_diversity"
        ),
    }


def get_topic_table(
    bundle: ArtifactBundle,
    include_outlier: bool = False,
) -> pd.DataFrame:
    """Trả bảng topic dùng trong trang tổng quan."""
    topics = bundle.topic_summary.copy()

    if not include_outlier:
        topics = topics[
            topics["topic_id"] != -1
        ]

    return topics.sort_values(
        by="document_count",
        ascending=False,
    ).reset_index(drop=True)


def get_documents_for_topic(
    bundle: ArtifactBundle,
    topic_id: int,
) -> pd.DataFrame:
    """Lấy danh sách bài báo thuộc một topic."""
    documents = bundle.document_topics

    selected = documents[
        documents["topic_id"] == topic_id
    ].copy()

    return selected.sort_values(
        by="topic_probability",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)


def get_keywords_for_topic(
    bundle: ArtifactBundle,
    topic_id: int,
) -> pd.DataFrame:
    """Lấy các từ khóa đại diện của một topic."""
    keywords = bundle.topic_keywords

    selected = keywords[
        keywords["topic_id"] == topic_id
    ].copy()

    return selected.sort_values(
        by="rank",
        ascending=True,
    ).reset_index(drop=True)