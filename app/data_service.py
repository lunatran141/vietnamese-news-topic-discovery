"""Đọc artifact của năm phương pháp cho giao diện Streamlit."""

from __future__ import annotations

import json
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


def _resolve_artifact_dir(
    bundle: ArtifactBundle,
    artifact_dir: str | Path | None = None,
) -> Path | None:
    if artifact_dir is not None:
        path = Path(artifact_dir)
        if path.is_dir():
            return path
    run_id = getattr(bundle.model_summary, "run_id", None)
    if run_id:
        for cand in ARTIFACT_ROOT.glob("*/*"):
            if cand.is_dir() and cand.name == run_id:
                return cand
    return None


def get_model_overview(
    bundle: ArtifactBundle,
    split: str | None = None,
    artifact_dir: str | Path | None = None,
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
    adjusted_rand_index = (
        summary.metrics.get(f"{metric_split}_adjusted_rand_index")
        or summary.metrics.get(f"{metric_split}_ari")
        or summary.metrics.get("adjusted_rand_index")
        or summary.metrics.get("ari")
    )
    normalized_mutual_info = (
        summary.metrics.get(f"{metric_split}_normalized_mutual_info")
        or summary.metrics.get(f"{metric_split}_nmi")
        or summary.metrics.get("normalized_mutual_info")
        or summary.metrics.get("nmi")
    )
    if metric_split == "test":
        coherence = (
            summary.metrics.get("test_npmi_coherence")
            or summary.metrics.get("test_npmi")
            or summary.metrics.get("npmi_coherence")
            or summary.metrics.get("npmi")
        )
    else:
        coherence = (
            summary.metrics.get("validation_npmi_coherence")
            or summary.metrics.get("validation_npmi")
            or summary.metrics.get("npmi_coherence")
            or summary.metrics.get("npmi")
        )

    fit_seconds = summary.metrics.get("fit_seconds")
    if fit_seconds is None:
        fit_seconds = getattr(summary, "fit_seconds", None)
    if fit_seconds is None and hasattr(summary, "model_extra") and summary.model_extra:
        fit_seconds = summary.model_extra.get("fit_seconds")

    representation_seconds = summary.metrics.get("representation_seconds")
    if representation_seconds is None:
        representation_seconds = getattr(summary, "representation_seconds", None)
    if representation_seconds is None and hasattr(summary, "model_extra") and summary.model_extra:
        representation_seconds = summary.model_extra.get("representation_seconds")

    val_predict = summary.metrics.get("validation_predict_seconds")
    if val_predict is None:
        val_predict = getattr(summary, "validation_inference_seconds", None)
    if val_predict is None and hasattr(summary, "model_extra") and summary.model_extra:
        val_predict = summary.model_extra.get("validation_inference_seconds")

    test_predict = summary.metrics.get("test_predict_seconds")
    if test_predict is None:
        test_predict = getattr(summary, "test_inference_seconds", None)
    if test_predict is None and hasattr(summary, "model_extra") and summary.model_extra:
        test_predict = summary.model_extra.get("test_inference_seconds")

    resolved_dir = _resolve_artifact_dir(bundle, artifact_dir)
    if resolved_dir:
        rt_file = resolved_dir / "runtime_summary.json"
        if rt_file.is_file():
            try:
                with rt_file.open("r", encoding="utf-8") as file:
                    rt_data = json.load(file)
                tv = rt_data.get("train_validation") if isinstance(rt_data, dict) else None
                test_rt = rt_data.get("test") if isinstance(rt_data, dict) else None
                if isinstance(tv, dict):
                    if fit_seconds is None:
                        u = tv.get("umap_fit_seconds")
                        h = tv.get("hdbscan_fit_seconds")
                        if u is not None or h is not None:
                            fit_seconds = (u or 0.0) + (h or 0.0)
                        else:
                            fit_seconds = tv.get("fit_seconds")
                    if val_predict is None:
                        val_predict = tv.get("validation_predict_seconds")
                if split == "test" and isinstance(test_rt, dict):
                    if representation_seconds is None and test_rt.get("representation_seconds") is not None:
                        representation_seconds = test_rt.get("representation_seconds")
                    if test_predict is None and test_rt.get("test_predict_seconds") is not None:
                        test_predict = test_rt.get("test_predict_seconds")
                if representation_seconds is None and isinstance(tv, dict):
                    representation_seconds = tv.get("representation_seconds") or tv.get("train_representation_seconds")
            except Exception:
                pass

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
        "fit_seconds": fit_seconds,
        "representation_seconds": representation_seconds,
        "validation_predict_seconds": val_predict,
        "test_predict_seconds": test_predict,
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
