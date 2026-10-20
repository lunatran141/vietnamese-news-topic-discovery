"""Đọc artifact của năm phương pháp cho giao diện Streamlit."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from src.artifacts.loader import ArtifactBundle, load_artifact_bundle


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path(
    os.getenv("TOPIC_ARTIFACT_ROOT", str(REPO_ROOT / "artifacts/common"))
)
DEFAULT_ARTIFACT_DIR = Path(
    os.getenv(
        "TOPIC_ARTIFACT_DIR",
        str(ARTIFACT_ROOT / "bertopic/e1_vietnamese_sbert"),
    )
)
METHOD_NAMES = {
    "bertopic": "BERTopic",
    "hdbscan": "HDBSCAN",
    "kmeans": "KMeans",
    "lda": "LDA",
    "nmf": "NMF",
}


@dataclass(frozen=True)
class ArtifactRun:
    method: str
    run_id: str
    path: Path

    @property
    def label(self) -> str:
        return f"{METHOD_NAMES.get(self.method, self.method)} · {self.run_id}"


def discover_artifact_runs(root: Path = ARTIFACT_ROOT) -> list[ArtifactRun]:
    """Tìm các run đã xuất manifest, không phụ thuộc tên thư mục cố định."""
    return sorted(
        (
            ArtifactRun(path.parent.parent.name, path.parent.name, path.parent)
            for path in root.glob("*/*/manifest.json")
        ),
        key=lambda run: (run.method, run.run_id),
    )


@st.cache_resource(show_spinner=False)
def load_dashboard_bundle(
    artifact_dir: str = str(DEFAULT_ARTIFACT_DIR),
) -> ArtifactBundle:
    return load_artifact_bundle(artifact_dir)


def get_available_splits(bundle: ArtifactBundle) -> list[str]:
    present = set(bundle.document_topics["split"].dropna().astype(str))
    return [split for split in ("train", "val", "test") if split in present]


def get_document_table(
    bundle: ArtifactBundle,
    split: str | None = None,
) -> pd.DataFrame:
    documents = bundle.document_topics
    if split and split != "all":
        documents = documents.loc[documents["split"] == split]
    return documents.copy()


def get_model_overview(
    bundle: ArtifactBundle,
    split: str | None = None,
) -> dict[str, int | float | str | None]:
    """Tính số lượng trên đúng split đang xem; metric run giữ nguyên."""
    summary = bundle.model_summary
    documents = get_document_table(bundle, split)
    outlier = documents["is_outlier"].fillna(False).astype(bool)
    outlier_count = int(outlier.sum())
    topic_count = int(documents.loc[~outlier, "topic_id"].nunique())

    return {
        "model_name": summary.model_name,
        "algorithm": summary.algorithm,
        "representation": summary.representation,
        "topic_count": topic_count,
        "document_count": len(documents),
        "assigned_document_count": len(documents) - outlier_count,
        "outlier_document_count": outlier_count,
        "outlier_rate": outlier_count / len(documents) if len(documents) else None,
        "runtime_seconds": summary.runtime_seconds,
        "topic_diversity": summary.metrics.get("topic_diversity"),
    }


def get_topic_table(
    bundle: ArtifactBundle,
    include_outlier: bool = False,
    split: str | None = None,
) -> pd.DataFrame:
    """Dùng assignments để tính lại size khi lọc split."""
    topics = bundle.topic_summary.copy()
    if split is not None:
        documents = get_document_table(bundle, split)
        counts = documents["topic_id"].value_counts()
        topics["document_count"] = topics["topic_id"].map(counts).fillna(0).astype(int)
        topics["percentage"] = (
            topics["document_count"] / len(documents) if len(documents) else 0.0
        )
        topics = topics.loc[topics["document_count"] > 0]
    if not include_outlier:
        topics = topics.loc[topics["topic_id"] != -1]
    return topics.sort_values("document_count", ascending=False).reset_index(drop=True)


def get_documents_for_topic(
    bundle: ArtifactBundle,
    topic_id: int,
    split: str | None = None,
) -> pd.DataFrame:
    documents = get_document_table(bundle, split)
    selected = documents.loc[documents["topic_id"] == topic_id].copy()
    if "distance_to_center" in selected:
        return selected.sort_values("distance_to_center").reset_index(drop=True)
    return selected.sort_values(
        "topic_probability", ascending=False, na_position="last"
    ).reset_index(drop=True)


def get_keywords_for_topic(bundle: ArtifactBundle, topic_id: int) -> pd.DataFrame:
    selected = bundle.topic_keywords.loc[
        bundle.topic_keywords["topic_id"] == topic_id
    ].copy()
    return selected.sort_values("rank").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_article_metadata() -> pd.DataFrame:
    """Bổ sung metadata chung cho artifact không xuất source/date."""
    path = REPO_ROOT / "data/processed/p1_normalized.csv"
    if not path.is_file():
        return pd.DataFrame(columns=["article_id"])
    columns = ["article_id", "title", "source", "published_at", "url"]
    return pd.read_csv(path, usecols=columns, encoding="utf-8-sig")


def enrich_documents(documents: pd.DataFrame) -> pd.DataFrame:
    metadata = load_article_metadata()
    if metadata.empty:
        return documents.copy()
    result = documents.merge(metadata, on="article_id", how="left", suffixes=("", "_meta"))
    for column in ("title", "source", "published_at", "url"):
        meta_column = f"{column}_meta"
        if meta_column in result:
            if column in result:
                result[column] = result[column].replace("", pd.NA).fillna(result[meta_column])
                result = result.drop(columns=meta_column)
            else:
                result = result.rename(columns={meta_column: column})
    return result


@st.cache_data(show_spinner=False)
def load_projection(artifact_dir: str) -> pd.DataFrame:
    """Đọc phép chiếu nếu run thật sự có file 2D."""
    root = Path(artifact_dir)
    path = root / "projection_2d.csv"
    if not path.is_file() and root.parent.name == "bertopic":
        path = REPO_ROOT / "results/bertopic_experiments" / root.name / "embedding_projection.csv"
    if not path.is_file():
        return pd.DataFrame()
    projection = pd.read_csv(path, encoding="utf-8-sig")
    if "topic" in projection and "topic_id" not in projection:
        projection = projection.rename(columns={"topic": "topic_id"})
    if not {"article_id", "x", "y", "topic_id"}.issubset(projection.columns):
        return pd.DataFrame()
    return projection


@st.cache_data(show_spinner=False)
def load_representatives(artifact_dir: str) -> pd.DataFrame:
    path = Path(artifact_dir) / "representative_docs.csv"
    return pd.read_csv(path, encoding="utf-8-sig") if path.is_file() else pd.DataFrame()


def get_representatives_for_topic(
    artifact_dir: str,
    topic_id: int,
    split: str = "all",
) -> pd.DataFrame:
    """Chỉ hiển thị bài đại diện thuộc topic và split đang xem."""
    representatives = load_representatives(artifact_dir)
    if representatives.empty:
        return representatives
    selected = representatives.loc[representatives["topic_id"] == topic_id]
    if split != "all" and "split" in selected:
        selected = selected.loc[selected["split"] == split]
    return selected.sort_values("rank").reset_index(drop=True)


def get_topic_trend(
    documents: pd.DataFrame,
    topic_id: int,
    frequency: str = "W",
) -> pd.DataFrame:
    if "published_at" not in documents:
        return pd.DataFrame(columns=["period", "count"])
    selected = documents.loc[documents["topic_id"] == topic_id].copy()
    selected["date"] = pd.to_datetime(
        selected["published_at"], errors="coerce", utc=True
    ).dt.tz_convert("Asia/Ho_Chi_Minh").dt.tz_localize(None)
    selected = selected.dropna(subset=["date"])
    if selected.empty:
        return pd.DataFrame(columns=["period", "count"])
    counts = selected.groupby(selected["date"].dt.to_period(frequency)).size()
    return pd.DataFrame({"period": counts.index.astype(str), "count": counts.values})
