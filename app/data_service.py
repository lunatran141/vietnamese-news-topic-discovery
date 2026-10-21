"""Đọc artifact của năm phương pháp cho giao diện Streamlit."""

from __future__ import annotations

import os
import re
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
    runs = [
        ArtifactRun(path.parent.parent.name, path.parent.name, path.parent)
        for path in root.glob("*/*/manifest.json")
    ]
    current_nmf = "nmf_p3_k10_seed42_dashboard_v2"
    if any(run.method == "nmf" and run.run_id == current_nmf for run in runs):
        runs = [
            run for run in runs
            if run.method != "nmf" or run.run_id == current_nmf
        ]
    return sorted(
        runs,
        key=lambda run: (run.method, run.run_id),
    )


def discover_model_runs(root: str | Path = ARTIFACT_ROOT) -> dict[str, str]:
    """Dạng mapping dùng bởi màn hình phân tích chi tiết NMF."""
    return {
        run.label: str(run.path)
        for run in discover_artifact_runs(Path(root))
    }


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
    supports_outlier = summary.algorithm.strip().lower() in {
        "bertopic",
        "hdbscan",
    }
    outlier = documents["is_outlier"].fillna(False).astype(bool)
    raw_outlier_count = int(outlier.sum())
    outlier_count = raw_outlier_count if supports_outlier else None
    topic_count = int(documents.loc[~outlier, "topic_id"].nunique())

    metric_split = "test" if split == "test" else "validation"
    adjusted_rand_index = summary.metrics.get(
        f"{metric_split}_adjusted_rand_index"
    )
    normalized_mutual_info = summary.metrics.get(
        f"{metric_split}_normalized_mutual_info"
    )
    if metric_split == "test":
        coherence = summary.metrics.get(
            "test_npmi_coherence",
            summary.metrics.get("npmi_coherence"),
        )
    else:
        coherence = summary.metrics.get(
            "validation_npmi_coherence",
            summary.metrics.get("npmi_coherence"),
        )

    return {
        "model_name": summary.model_name,
        "algorithm": summary.algorithm,
        "representation": summary.representation,
        "topic_count": topic_count,
        "document_count": len(documents),
        "assigned_document_count": len(documents) - raw_outlier_count,
        "outlier_document_count": outlier_count,
        "outlier_rate": (
            raw_outlier_count / len(documents)
            if supports_outlier and len(documents)
            else None
        ),
        "supports_outlier": supports_outlier,
        "runtime_seconds": summary.runtime_seconds,
        "runtime_scope": getattr(summary, "runtime_scope", None),
        "fit_seconds": summary.metrics.get("fit_seconds"),
        "representation_seconds": summary.metrics.get(
            "representation_seconds"
        ),
        "validation_predict_seconds": summary.metrics.get(
            "validation_predict_seconds"
        ),
        "test_predict_seconds": summary.metrics.get(
            "test_predict_seconds"
        ),
        "adjusted_rand_index": adjusted_rand_index,
        "normalized_mutual_info": normalized_mutual_info,
        "coherence": coherence,
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


def filter_documents(
    bundle: ArtifactBundle,
    split: str | None = None,
    multi_topic_only: bool = False,
    gap_max: float | None = None,
    query: str = "",
) -> pd.DataFrame:
    """Lọc bài cho phân tích NMF/HDBSCAN mà không đổi artifact gốc."""
    frame = bundle.document_topics.copy()
    if split and split != "all":
        frame = frame.loc[frame["split"].eq(split)]
    if multi_topic_only:
        if "is_multi_topic" not in frame:
            return frame.iloc[:0]
        flags = frame["is_multi_topic"].astype(str).str.lower().eq("true")
        frame = frame.loc[flags]
    if gap_max is not None:
        if "top_two_share_gap" not in frame:
            return frame.iloc[:0]
        gap = pd.to_numeric(frame["top_two_share_gap"], errors="coerce")
        frame = frame.loc[gap.le(gap_max)]
    if query:
        matches = frame["article_id"].astype(str).str.contains(
            query, case=False, regex=False
        )
        if "title" in frame:
            matches |= frame["title"].fillna("").astype(str).str.contains(
                query, case=False, regex=False
            )
        frame = frame.loc[matches]
    return frame.reset_index(drop=True)


def get_article_distribution(
    bundle: ArtifactBundle,
    article_id: str,
) -> pd.DataFrame:
    """Lấy toàn bộ topic share của một bài NMF."""
    rows = bundle.document_topics.loc[
        bundle.document_topics["article_id"].eq(article_id)
    ]
    if rows.empty:
        return pd.DataFrame(columns=["topic_id", "topic_share"])
    values = [
        {
            "topic_id": int(column.split("_")[1]),
            "topic_share": float(rows.iloc[0][column]),
        }
        for column in rows.columns
        if re.fullmatch(r"topic_\d+_share", column)
    ]
    return pd.DataFrame(values, columns=["topic_id", "topic_share"]).sort_values(
        ["topic_share", "topic_id"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)


def get_split_metrics(bundle: ArtifactBundle) -> pd.DataFrame:
    """Trả metric validation/test đúng phạm vi đã công bố."""
    metrics = bundle.model_summary.metrics
    rows = []
    for split in ("validation", "test"):
        row = {
            "split": split,
            "fit_split": "train" if split == "validation" else "train+val",
        }
        for label, legacy in (
            ("ari", "adjusted_rand_index"),
            ("nmi", "normalized_mutual_info"),
            ("npmi", "npmi_coherence"),
            ("topic_diversity", "topic_diversity"),
            ("reconstruction_error", "reconstruction_error"),
            ("normalized_reconstruction_error", "normalized_reconstruction_error"),
        ):
            row[label] = metrics.get(f"{split}_{label}")
            if (
                row[label] is None
                and split == "test"
                and bundle.model_summary.algorithm.upper() in {"NMF", "LDA"}
            ):
                row[label] = metrics.get(legacy)
        if any(
            value is not None
            for key, value in row.items()
            if key not in {"split", "fit_split"}
        ):
            rows.append(row)
    return pd.DataFrame(rows)


def load_optional_table(
    artifact_dir: str | Path,
    filename: str,
) -> pd.DataFrame:
    """Đọc bảng phụ cho giao diện, giới hạn ở tên file đã biết."""
    if filename not in {"representative_docs.csv", "projection_2d.csv"}:
        raise ValueError("Bảng phụ không được hỗ trợ")
    path = Path(artifact_dir) / filename
    return (
        pd.read_csv(path, encoding="utf-8-sig")
        if path.is_file()
        else pd.DataFrame()
    )
