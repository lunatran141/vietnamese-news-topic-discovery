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


def _require_files(
    paths: list[Path],
    model_label: str = "BERTopic",
) -> None:
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
            f"Thiếu file {model_label} đầu vào: "
            f"{missing_text}"
        )


def _compute_topic_diversity(
    topic_keywords: pd.DataFrame,
    top_n: int = 10,
) -> float | None:
    """Tính độ đa dạng từ khóa giữa các topic trên top-N từ khóa đầu."""
    if topic_keywords.empty:
        return None

    top_rows = topic_keywords[
        topic_keywords["rank"] <= top_n
    ]
    keywords = [
        str(word).strip()
        for word in top_rows["keyword"].tolist()
        if str(word).strip()
    ]
    if not keywords:
        return None

    return round(len(set(keywords)) / len(keywords), 4)


def _build_kmeans_topic_summary(
    document_topics: pd.DataFrame,
    topic_keywords: pd.DataFrame,
    representative_titles: int = 3,
) -> pd.DataFrame:
    """Tạo bảng topic_summary.csv cho kết quả phân cụm K-Means."""
    total_documents = len(document_topics)
    topic_ids = sorted(
        set(
            document_topics["topic_id"]
            .dropna()
            .astype(int)
            .tolist()
        )
        | set(
            topic_keywords["topic_id"]
            .dropna()
            .astype(int)
            .tolist()
        )
    )

    rows: list[dict[str, Any]] = []
    for topic_id in topic_ids:
        if topic_id == -1:
            continue

        group = document_topics[
            document_topics["topic_id"] == topic_id
        ]
        topic_words = (
            topic_keywords[
                topic_keywords["topic_id"] == topic_id
            ]
            .sort_values("rank", ascending=True)
            .head(5)["keyword"]
            .astype(str)
            .tolist()
        )

        if topic_words:
            topic_name = (
                f"{int(topic_id)}_"
                + "_".join(topic_words[:3])
            )
        else:
            topic_name = f"topic_{int(topic_id)}"

        doc_count = int(len(group))
        percentage = (
            doc_count / total_documents
            if total_documents > 0
            else 0.0
        )

        row: dict[str, Any] = {
            "topic_id": int(topic_id),
            "topic_name": topic_name,
            "document_count": doc_count,
            "percentage": percentage,
            "top_keywords": ", ".join(topic_words),
        }

        if "title" in group.columns:
            ordered_group = group
            if "distance_to_center" in group.columns:
                ordered_group = group.sort_values(
                    "distance_to_center",
                    ascending=True,
                )
            titles = (
                ordered_group["title"]
                .fillna("")
                .astype(str)
            )
            titles = (
                titles[titles != ""]
                .head(representative_titles)
                .tolist()
            )
            row["representative_titles"] = " | ".join(
                titles
            )

        if "category_gold" in group.columns:
            categories = (
                group["category_gold"]
                .fillna("")
                .astype(str)
            )
            categories = categories[categories != ""]
            if not categories.empty:
                category_counts = categories.value_counts()
                row["dominant_category"] = str(
                    category_counts.index[0]
                )
                row["category_purity"] = float(
                    category_counts.iloc[0]
                    / category_counts.sum()
                )
            else:
                row["dominant_category"] = ""
                row["category_purity"] = None

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return pd.DataFrame(
            columns=[
                "topic_id",
                "topic_name",
                "document_count",
                "percentage",
                "top_keywords",
            ]
        )

    return summary_df.sort_values(
        by=["document_count", "topic_id"],
        ascending=[False, True],
    ).reset_index(drop=True)


def export_kmeans_artifacts(
    source_dir: str | Path,
    output_dir: str | Path,
    run_id: str | None = None,
) -> ArtifactBundle:
    """Chuyển kết quả K-Means sang artifact schema chung."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    document_topics_path = (
        source_dir / "document_topics.csv"
    )
    topic_keywords_path = (
        source_dir / "topic_keywords.json"
    )
    run_metadata_path = (
        source_dir / "run_metadata.json"
    )
    config_path = source_dir / "config.json"

    _require_files(
        [
            document_topics_path,
            topic_keywords_path,
            run_metadata_path,
            config_path,
        ],
        model_label="KMeans",
    )

    raw_documents = pd.read_csv(
        document_topics_path,
        encoding="utf-8-sig",
    )
    raw_keywords = _read_json(topic_keywords_path)
    run_metadata = _read_json(run_metadata_path)
    config = _read_json(config_path)

    document_topics = raw_documents.rename(
        columns={
            "cluster_id": "topic_id",
        }
    ).copy()

    if "topic_id" not in document_topics.columns:
        raise ValueError(
            "document_topics.csv của KMeans phải có cột "
            "cluster_id hoặc topic_id"
        )

    document_topics["topic_id"] = document_topics[
        "topic_id"
    ].astype(int)
    document_topics["topic_probability"] = None
    document_topics["is_outlier"] = False

    if "title" not in document_topics.columns:
        document_topics["title"] = ""

    base_columns = [
        "article_id",
        "split",
        "topic_id",
        "topic_probability",
        "is_outlier",
        "title",
    ]
    extra_columns = [
        col
        for col in document_topics.columns
        if col not in base_columns
    ]
    document_topics = document_topics[
        base_columns + extra_columns
    ]

    keyword_rows: list[dict[str, Any]] = []
    sorted_cluster_keys = sorted(
        raw_keywords.keys(),
        key=lambda key: int(key),
    )
    for cluster_key in sorted_cluster_keys:
        topic_id = int(cluster_key)
        items = raw_keywords[cluster_key]
        for rank, item in enumerate(items, start=1):
            keyword_rows.append(
                {
                    "topic_id": topic_id,
                    "rank": rank,
                    "keyword": str(
                        item.get(
                            "term",
                            item.get("keyword", ""),
                        )
                    ),
                    "weight": float(item["weight"]),
                }
            )

    topic_keywords = pd.DataFrame(
        keyword_rows,
        columns=[
            "topic_id",
            "rank",
            "keyword",
            "weight",
        ],
    )

    topic_summary = _build_kmeans_topic_summary(
        document_topics=document_topics,
        topic_keywords=topic_keywords,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    topic_summary.to_csv(
        output_dir / "topic_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    document_topics.to_csv(
        output_dir / "document_topics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    topic_keywords.to_csv(
        output_dir / "topic_keywords.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (output_dir / "config_used.yaml").open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            config,
            file,
            allow_unicode=True,
            sort_keys=False,
        )

    resolved_run_id = str(
        run_id
        or output_dir.name
        or run_metadata.get(
            "pipeline_name",
            "kmeans_baseline",
        )
    )

    metrics_validation = run_metadata.get(
        "metrics_validation",
        {},
    )
    train_count = int(
        (document_topics["split"] == "train").sum()
    )
    validation_count = int(
        document_topics["split"]
        .isin(["val", "validation"])
        .sum()
    )
    topic_diversity = _compute_topic_diversity(
        topic_keywords
    )

    model_summary: dict[str, Any] = {
        "model_name": f"KMeans {resolved_run_id}",
        "run_id": resolved_run_id,
        "algorithm": "KMeans",
        "representation": "TF-IDF",
        "topic_count": int(
            run_metadata.get(
                "n_clusters",
                len(topic_summary),
            )
        ),
        "outlier_rate": 0.0,
        "runtime_seconds": run_metadata.get(
            "execution_duration_seconds"
        ),
        "metrics": {
            "topic_diversity": topic_diversity,
            "train_outlier_rate": 0.0,
            "validation_outlier_rate": 0.0,
            "train_count": train_count,
            "validation_count": validation_count,
            **metrics_validation,
        },
        "config": config,
    }

    manifest = {
        "schema_version": "1.0",
        "model_name": model_summary["model_name"],
        "run_id": resolved_run_id,
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