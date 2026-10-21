"""Cung cấp dữ liệu artifact cho giao diện Streamlit."""

from __future__ import annotations

import os
import json
import re
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

    score = "topic_share" if "topic_share" in selected else "topic_probability"
    return selected.sort_values(
        by=score,
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)


def discover_model_runs(root: str | Path = "artifacts/common") -> dict[str, str]:
    """Liệt kê mô hình, chỉ hiển thị run NMF đã bàn giao cho dashboard."""
    runs = []
    current_nmf = "nmf_p3_k10_seed42_dashboard_v2"
    has_current_nmf = (Path(root) / "nmf" / current_nmf / "manifest.json").is_file()
    for manifest in Path(root).glob("*/*/manifest.json"):
        with manifest.open(encoding="utf-8") as source:
            item = json.load(source)
        algorithm = manifest.parent.parent.name.upper()
        if algorithm == "NMF" and has_current_nmf and item["run_id"] != current_nmf:
            continue
        label = f"{algorithm} · {item['run_id']}"
        priority = 0 if algorithm == "NMF" and "dashboard" in item["run_id"] else 1
        runs.append((priority, label, str(manifest.parent)))
    return {label: path for _, label, path in sorted(runs)}


def filter_documents(
    bundle: ArtifactBundle, split: str | None = None, multi_topic_only: bool = False,
    gap_max: float | None = None, query: str = "",
) -> pd.DataFrame:
    """Lọc bản sao dữ liệu; gap không thay đổi cờ đa chủ đề đã công bố."""
    frame = bundle.document_topics.copy()
    if split is not None:
        frame = frame.loc[frame["split"].eq(split)]
    if multi_topic_only:
        if "is_multi_topic" not in frame:
            return frame.iloc[:0]
        flags = frame["is_multi_topic"].astype(str).str.lower().eq("true")
        frame = frame.loc[flags]
    if gap_max is not None:
        if "top_two_share_gap" not in frame:
            return frame.iloc[:0]
        frame = frame.loc[frame["top_two_share_gap"].le(gap_max)]
    if query:
        matches = frame["article_id"].astype(str).str.contains(query, case=False, regex=False)
        if "title" in frame:
            matches |= frame["title"].fillna("").str.contains(query, case=False, regex=False)
        frame = frame.loc[matches]
    return frame.reset_index(drop=True)


def get_article_distribution(bundle: ArtifactBundle, article_id: str) -> pd.DataFrame:
    """Lấy toàn bộ topic share của một bài, không gọi đó là xác suất NMF."""
    rows = bundle.document_topics.loc[bundle.document_topics["article_id"].eq(article_id)]
    if rows.empty:
        return pd.DataFrame(columns=["topic_id", "topic_share"])
    values = [
        {"topic_id": int(column.split("_")[1]), "topic_share": float(rows.iloc[0][column])}
        for column in rows.columns if re.fullmatch(r"topic_\d+_share", column)
    ]
    return pd.DataFrame(values, columns=["topic_id", "topic_share"]).sort_values(
        ["topic_share", "topic_id"], ascending=[False, True], kind="stable",
    ).reset_index(drop=True)


def get_split_metrics(bundle: ArtifactBundle) -> pd.DataFrame:
    """Mỗi hàng ghi rõ split và nguồn mô hình; không suy ra metric còn thiếu."""
    metrics = bundle.model_summary.metrics
    rows = []
    for split in ("validation", "test"):
        row = {"split": split, "fit_split": "train" if split == "validation" else "train+val"}
        for label, legacy in (
            ("ari", "adjusted_rand_index"), ("nmi", "normalized_mutual_info"),
            ("npmi", "npmi_coherence"), ("topic_diversity", "topic_diversity"),
            ("reconstruction_error", "reconstruction_error"),
            ("normalized_reconstruction_error", "normalized_reconstruction_error"),
        ):
            row[label] = metrics.get(f"{split}_{label}")
            if row[label] is None and split == "test" and bundle.model_summary.algorithm.upper() in {"NMF", "LDA"}:
                row[label] = metrics.get(legacy)
        if any(row[key] is not None for key in row if key not in {"split", "fit_split"}):
            rows.append(row)
    return pd.DataFrame(rows)


def load_optional_table(artifact_dir: str | Path, filename: str) -> pd.DataFrame:
    """Chỉ đọc hai bảng phụ đã biết; không nhận đường dẫn tùy ý từ giao diện."""
    if filename not in {"representative_docs.csv", "projection_2d.csv"}:
        raise ValueError("Bảng phụ không được hỗ trợ")
    path = Path(artifact_dir) / filename
    return pd.read_csv(path, encoding="utf-8-sig") if path.is_file() else pd.DataFrame()


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
