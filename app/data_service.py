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


def _as_boolean_series(series: pd.Series) -> pd.Series:
    """Chuẩn hóa cột boolean đọc từ CSV."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def _resolve_linked_file(
    artifact_dir: str | Path,
    value: str | Path,
) -> Path:
    """Tìm file được khai báo bằng đường dẫn tương đối trong metadata."""

    linked = Path(value)

    if linked.is_absolute():
        return linked

    artifact_path = Path(artifact_dir).resolve()

    search_roots = [
        Path.cwd(),
        artifact_path,
        *artifact_path.parents,
    ]

    for root in search_roots:
        candidate = root / linked
        if candidate.is_file():
            return candidate

    return Path.cwd() / linked


def load_hdbscan_run_metadata(
    artifact_dir: str | Path,
) -> dict:
    """Đọc run_metadata.json của HDBSCAN."""

    path = Path(artifact_dir) / "run_metadata.json"

    if not path.is_file():
        return {}

    return json.loads(
        path.read_text(encoding="utf-8-sig")
    )


def load_hdbscan_runtime(
    artifact_dir: str | Path,
) -> dict:
    """Đọc runtime_summary.json mà không tạo số thay thế."""

    path = Path(artifact_dir) / "runtime_summary.json"

    if not path.is_file():
        return {}

    return json.loads(
        path.read_text(encoding="utf-8-sig")
    )


def get_hdbscan_split_summary(
    bundle: ArtifactBundle,
) -> pd.DataFrame:
    """Tính coverage và outlier theo split từ document_topics.csv."""

    documents = bundle.document_topics.copy()
    documents["is_outlier"] = _as_boolean_series(
        documents["is_outlier"]
    )

    rows = []

    for split, group in documents.groupby(
        "split",
        sort=False,
    ):
        document_count = int(len(group))
        outlier_count = int(group["is_outlier"].sum())
        assigned_count = document_count - outlier_count

        rows.append(
            {
                "split": split,
                "document_count": document_count,
                "assigned_count": assigned_count,
                "outlier_count": outlier_count,
                "coverage": (
                    assigned_count / document_count
                    if document_count
                    else None
                ),
                "outlier_rate": (
                    outlier_count / document_count
                    if document_count
                    else None
                ),
            }
        )

    return pd.DataFrame(rows)


def load_hdbscan_stability(
    artifact_dir: str | Path,
) -> pd.DataFrame:
    """Đọc stability summary theo đường dẫn được khai báo."""

    metadata = load_hdbscan_run_metadata(artifact_dir)

    value = metadata.get("stability_summary_path")

    if not value:
        value = (
            metadata.get("source_files", {})
            .get("stability_summary")
        )

    if not value:
        return pd.DataFrame()

    path = _resolve_linked_file(
        artifact_dir,
        value,
    )

    if not path.is_file():
        return pd.DataFrame()

    return pd.read_csv(
        path,
        encoding="utf-8-sig",
    )


def load_hdbscan_projection(
    bundle: ArtifactBundle,
    artifact_dir: str | Path,
) -> pd.DataFrame:
    """Ghép tọa độ 2D với metadata phục vụ tooltip."""

    projection_path = (
        Path(artifact_dir) / "projection_2d.csv"
    )

    if not projection_path.is_file():
        return pd.DataFrame()

    projection = pd.read_csv(
        projection_path,
        encoding="utf-8-sig",
        dtype={"article_id": str},
    )

    documents = bundle.document_topics.copy()
    documents["article_id"] = (
        documents["article_id"].astype(str)
    )

    document_columns = [
        column
        for column in (
            "article_id",
            "title",
            "membership_strength",
            "soft_primary_topic_id",
            "soft_primary_membership_strength",
            "second_topic_id",
            "second_membership_strength",
            "top_two_share_gap",
            "is_multi_topic",
            "is_outlier",
            "category_gold",
        )
        if column in documents.columns
    ]

    projection = projection.merge(
        documents[document_columns],
        on="article_id",
        how="left",
        validate="one_to_one",
    )

    metadata = load_hdbscan_run_metadata(
        artifact_dir
    )

    manifest_value = (
        metadata.get("source_files", {})
        .get(
            "split_manifest",
            "data/splits/split_manifest.csv",
        )
    )

    manifest_path = _resolve_linked_file(
        artifact_dir,
        manifest_value,
    )

    if manifest_path.is_file():
        manifest = pd.read_csv(
            manifest_path,
            encoding="utf-8-sig",
            dtype={"article_id": str},
        )

        metadata_columns = [
            column
            for column in (
                "article_id",
                "source",
                "published_at",
            )
            if column in manifest.columns
        ]

        projection = projection.merge(
            manifest[metadata_columns],
            on="article_id",
            how="left",
            validate="one_to_one",
        )

    if "source" not in projection.columns:
        projection["source"] = None

    if "published_at" not in projection.columns:
        projection["published_at"] = None

    projection["published_at"] = pd.to_datetime(
        projection["published_at"],
        errors="coerce",
    )

    projection["is_outlier"] = _as_boolean_series(
        projection["is_outlier"]
    )

    projection["topic_label"] = projection[
        "topic_id"
    ].map(
        lambda value: (
            "Outlier (-1)"
            if int(value) == -1
            else f"Topic {int(value)}"
        )
    )

    return projection


def load_hdbscan_human_reviews(
    artifact_dir: str | Path,
) -> pd.DataFrame:
    """Đọc kết quả human evaluation nếu đã hoàn thành."""

    relative_path = Path(
        "results/hdbscan/human_eval/normalized/"
        "human_eval_reviews.csv"
    )

    path = _resolve_linked_file(
        artifact_dir,
        relative_path,
    )

    if not path.is_file():
        return pd.DataFrame()

    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"article_id": str},
    )

def load_hdbscan_experiment_comparison(
    artifact_dir: str | Path,
) -> pd.DataFrame:
    """Đọc kết quả so sánh MiniLM/MPNet và EOM/Leaf."""

    summary_path = _resolve_linked_file(
        artifact_dir,
        "results/hdbscan/m11_summary.csv",
    )
    validation_path = _resolve_linked_file(
        artifact_dir,
        (
            "results/hdbscan/validation/"
            "m11_validation_metrics.csv"
        ),
    )

    if (
        not summary_path.is_file()
        or not validation_path.is_file()
    ):
        return pd.DataFrame()

    train_summary = pd.read_csv(
        summary_path,
        encoding="utf-8-sig",
    )
    validation = pd.read_csv(
        validation_path,
        encoding="utf-8-sig",
    )

    target_embeddings = {
        "minilm_title_content",
        "mpnet_title_content",
    }
    target_experiments = {
        "eom_15_5",
        "leaf_15_5",
    }

    train_summary = train_summary.loc[
        train_summary["embedding_run"].isin(
            target_embeddings
        )
        & train_summary["experiment"].isin(
            target_experiments
        )
    ].copy()

    validation = validation.loc[
        validation["embedding_run"].isin(
            target_embeddings
        )
        & validation["experiment"].isin(
            target_experiments
        )
        & validation["split"].eq("val")
    ].copy()

    merged = validation.merge(
        train_summary,
        on=[
            "embedding_run",
            "experiment",
            "config_hash",
        ],
        how="inner",
        validate="one_to_one",
        suffixes=("_validation", "_train"),
    )

    if merged.empty:
        return pd.DataFrame()

    embedding_names = {
        "minilm_title_content": "MiniLM",
        "mpnet_title_content": "MPNet",
    }

    result = pd.DataFrame(
        {
            "embedding": merged[
                "embedding_run"
            ].map(embedding_names),
            "embedding_run": merged[
                "embedding_run"
            ],
            "selection_method": merged[
                "cluster_selection_method"
            ].astype(str).str.upper(),
            "experiment": merged["experiment"],
            "min_cluster_size": merged[
                "min_cluster_size"
            ],
            "min_samples": merged[
                "min_samples"
            ],
            "train_count": merged[
                "n_articles_train"
            ],
            "topic_count": merged["n_clusters"],
            "train_outlier_count": merged[
                "n_noise_train"
            ],
            "train_outlier_rate": (
                merged["noise_percent_train"] / 100
            ),
            "validation_count": merged[
                "n_articles_validation"
            ],
            "validation_outlier_count": merged[
                "n_noise_validation"
            ],
            "validation_outlier_rate": (
                merged[
                    "noise_percent_validation"
                ]
                / 100
            ),
            "validation_coverage": (
                merged["coverage_percent"] / 100
            ),
            "validation_ari": merged["ari"],
            "validation_nmi": merged["nmi"],
        }
    )

    result["configuration"] = (
        result["embedding"]
        + " · "
        + result["selection_method"]
    )

    return result.sort_values(
        [
            "embedding",
            "selection_method",
        ],
        kind="stable",
    ).reset_index(drop=True)


def load_hdbscan_error_cases(
    artifact_dir: str | Path,
) -> pd.DataFrame:
    """Đọc các hard cases đã được phân tích thủ công."""

    path = _resolve_linked_file(
        artifact_dir,
        (
            "results/hdbscan/error_analysis/"
            "hdbscan_error_cases.csv"
        ),
    )

    if not path.is_file():
        return pd.DataFrame()

    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"article_id": str},
    )


def load_hdbscan_transition_hard_cases(
    bundle: ArtifactBundle,
    artifact_dir: str | Path,
) -> pd.DataFrame:
    """Tổng hợp bài đổi trạng thái cluster/outlier giữa các seed."""

    metadata = load_hdbscan_run_metadata(
        artifact_dir
    )

    stability_value = metadata.get(
        "stability_summary_path"
    )

    if not stability_value:
        stability_value = (
            metadata.get("source_files", {})
            .get("stability_summary")
        )

    if not stability_value:
        return pd.DataFrame()

    stability_summary_path = _resolve_linked_file(
        artifact_dir,
        stability_value,
    )

    stability_dir = stability_summary_path.parent

    transition_paths = [
        stability_dir / "train_transition_cases.csv",
        stability_dir / "val_transition_cases.csv",
    ]

    frames = []

    for path in transition_paths:
        if path.is_file():
            frames.append(
                pd.read_csv(
                    path,
                    encoding="utf-8-sig",
                    dtype={"article_id": str},
                )
            )

    if not frames:
        return pd.DataFrame()

    transitions = pd.concat(
        frames,
        ignore_index=True,
    )

    identity_columns = [
        "split",
        "article_id",
    ]

    counts = (
        transitions.groupby(
            identity_columns,
            as_index=False,
        )
        .size()
        .rename(
            columns={"size": "transition_count"}
        )
    )

    direction_counts = (
        transitions.pivot_table(
            index=identity_columns,
            columns="direction",
            values="seed_a",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )

    direction_counts.columns.name = None

    result = counts.merge(
        direction_counts,
        on=identity_columns,
        how="left",
        validate="one_to_one",
    )

    for column in (
        "noise_to_cluster",
        "cluster_to_noise",
    ):
        if column not in result.columns:
            result[column] = 0

    seeds = sorted(
        set(transitions["seed_a"].tolist())
        | set(transitions["seed_b"].tolist())
    )

    pair_count = (
        len(seeds) * (len(seeds) - 1) // 2
    )

    result["compared_seed_pairs"] = pair_count

    result["instability_rate"] = (
        result["transition_count"] / pair_count
        if pair_count
        else None
    )

    documents = bundle.document_topics.copy()
    documents["article_id"] = (
        documents["article_id"].astype(str)
    )

    document_columns = [
        column
        for column in (
            "article_id",
            "title",
            "topic_id",
            "membership_strength",
            "soft_primary_topic_id",
            "second_topic_id",
            "top_two_share_gap",
            "is_multi_topic",
            "is_outlier",
            "category_gold",
        )
        if column in documents.columns
    ]

    result = result.merge(
        documents[document_columns],
        on="article_id",
        how="left",
        validate="many_to_one",
    )

    return result.sort_values(
        [
            "transition_count",
            "article_id",
        ],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)