"""Chuyển artifact của từng mô hình sang schema chung."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import yaml

from src.artifacts.loader import (
    ArtifactBundle,
    load_artifact_bundle,
)
from src.artifacts.nmf_parquet import (
    normalize_nmf_topic_id,
    read_nmf_parquet_assignments,
)
from src.data.dataset_loader import load_dataset
from src.features.article_text import build_article_text
from src.models.lda_pipeline import transform_lda_pipeline

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


def export_lda_artifacts(
    source_dir: str | Path = "artifacts/lda/final",
    output_dir: str | Path = "artifacts/common/lda/lda_k15_seed42",
    dataset_path: str | Path = "data/processed/p2_segmented.csv",
    split_manifest_path: str | Path = "data/splits/split_manifest.csv",
    final_metrics_path: str | Path = "results/lda/final/test_metrics.json",
    representatives_path: str | Path = (
        "results/lda/final/representative_docs_development.csv"
    ),
) -> ArtifactBundle:
    """Export frozen LDA K=15 into the shared artifact schema."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}"
        )

    _require_files(
        [
            source_dir / "lda_pipeline.joblib",
            source_dir / "run_metadata.json",
            source_dir / "config.json",
            Path(dataset_path),
            Path(split_manifest_path),
            Path(final_metrics_path),
            Path(representatives_path),
        ],
        model_label="LDA Final",
    )

    metadata = _read_json(source_dir / "run_metadata.json")
    final_config = _read_json(source_dir / "config.json")

    final_metrics = _read_json(Path(final_metrics_path))

    if int(final_metrics["n_topics"]) != 15:
        raise ValueError("Final Test metrics K mismatch")

    if int(final_metrics["test_documents"]) != 1390:
        raise ValueError("Final Test document count mismatch")

    if final_metrics["npmi_definition"] != "document_binary_common":
        raise ValueError("Unexpected NPMI definition")

    pipeline = joblib.load(source_dir / "lda_pipeline.joblib")
    model = pipeline["lda_model"]
    vectorizer = pipeline["vectorizer"]

    k = int(metadata["selected_n_topics"])
    seed = int(metadata["random_state"])

    if k != 15 or model.n_components != 15 or seed != 42:
        raise ValueError("Expected frozen LDA K15 seed42")

    if metadata["dataset_version"] != "v1.0":
        raise ValueError("Dataset version mismatch")
    if metadata["preprocessing_variant"] != "p2_segmented":
        raise ValueError("Preprocessing mismatch")
    if metadata["fit_split"] != "train+val":
        raise ValueError("Fit split mismatch")

    dataset = load_dataset(dataset_path)
    splits = pd.read_csv(
        split_manifest_path,
        encoding="utf-8-sig",
    )

    if not {"article_id", "split"}.issubset(splits.columns):
        raise ValueError("Invalid split manifest")

    if dataset["article_id"].duplicated().any():
        raise ValueError("Duplicate dataset IDs")
    if splits["article_id"].duplicated().any():
        raise ValueError("Duplicate split IDs")

    if len(dataset) != 9274 or len(splits) != 9274:
        raise ValueError("Expected Dataset v1.0: 9274 rows")

    dataset = dataset.copy()
    splits = splits.copy()

    dataset["article_id"] = dataset["article_id"].astype(str)
    splits["article_id"] = splits["article_id"].astype(str)

    if set(dataset["article_id"]) != set(splits["article_id"]):
        raise ValueError("Dataset and split IDs mismatch")

    if set(splits["split"]) != {"train", "val", "test"}:
        raise ValueError("Invalid split names")

    dataset = dataset.merge(
        splits[["article_id", "split"]],
        on="article_id",
        how="left",
        validate="one_to_one",
    )

    if dataset["split"].isna().any():
        raise ValueError("Missing split assignment")

    texts = build_article_text(dataset, "text").tolist()
    bow = vectorizer.transform(texts)
    probabilities = transform_lda_pipeline(model, bow)

    if probabilities.shape != (len(dataset), k):
        raise ValueError("Invalid LDA distribution shape")


    # Main and second topic probabilities from frozen LDA K15.
    sorted_topic_indices = np.argsort(
        probabilities, axis=1
    )[:, ::-1]

    topic_ids = sorted_topic_indices[:, 0]
    second_topic_ids = sorted_topic_indices[:, 1]

    row_indices = np.arange(len(dataset))

    topic_probs = probabilities[row_indices, topic_ids]
    second_topic_probs = probabilities[
        row_indices, second_topic_ids
    ]

    # Difference between first and second topic probabilities.
    top_two_share_gap = topic_probs - second_topic_probs

    # Shannon entropy, using natural logarithm.
    # Clip only for numerical stability in log().
    safe_probs = np.clip(probabilities, 1e-12, 1.0)
    topic_entropy = -np.sum(
        probabilities * np.log(safe_probs), axis=1
    )

    if not np.isfinite(topic_entropy).all():
        raise ValueError("Invalid LDA topic entropy")

    if (top_two_share_gap < -1e-10).any():
        raise ValueError("Invalid top-two topic probability gap")



    # PCA projection for visualization only.
    # Fit on Train + Validation to keep Test held out.
    development_mask = dataset["split"].isin(
        ["train", "val"]
    ).to_numpy()

    if development_mask.sum() != 7884:
        raise ValueError("Invalid development split count")

    pca = PCA(n_components=2, svd_solver="full")

    pca.fit(probabilities[development_mask])

    coordinates = pca.transform(probabilities)

    if coordinates.shape != (len(dataset), 2):
        raise ValueError("Invalid projection shape")

    if not np.isfinite(coordinates).all():
        raise ValueError("Non-finite projection coordinates")

    projection_2d = pd.DataFrame({
        "article_id": dataset["article_id"].to_numpy(),
        "split": dataset["split"].to_numpy(),
        "topic_id": topic_ids.astype(int),
        "x": coordinates[:, 0],
        "y": coordinates[:, 1],
    })


    if not np.isfinite(topic_probs).all():
        raise ValueError("Invalid topic probabilities")
    if ((topic_probs < 0) | (topic_probs > 1)).any():
        raise ValueError("Topic probabilities outside [0,1]")


    # Additional LDA probability fields required by the leader.
    document_topics = dataset[
        [
            "article_id",
            "split",
            "title",
            "source",
            "published_at",
            "category_gold",
        ]
    ].copy()

    document_topics["topic_id"] = topic_ids.astype(int)
    document_topics["topic_probability"] = topic_probs

    document_topics["second_topic_id"] = (
        second_topic_ids.astype(int)
    )

    document_topics["second_topic_probability"] = (
        second_topic_probs
    )

    document_topics["top_two_share_gap"] = (
        top_two_share_gap
    )

    document_topics["topic_entropy"] = topic_entropy

    # Leader-approved multi-topic threshold.
    # A document is multi-topic if its second topic
    # has a probability of at least 0.20.
    MULTI_TOPIC_THRESHOLD = 0.20

    document_topics["is_multi_topic"] = (
        document_topics["second_topic_probability"]
        >= MULTI_TOPIC_THRESHOLD
    )

    # LDA has no explicit outlier label.
    document_topics["is_outlier"] = False

    document_topics = document_topics[
        [
            "article_id",
            "split",
            "topic_id",
            "topic_probability",
            "second_topic_id",
            "second_topic_probability",
            "top_two_share_gap",
            "topic_entropy",
            "is_multi_topic",
            "title",
            "category_gold",
            "is_outlier",
            "source",
            "published_at",
        ]
    ]


    feature_names = vectorizer.get_feature_names_out()
    keyword_rows = []

    for topic_id, weights in enumerate(model.components_):
        indices = np.argsort(weights)[::-1][:10]

        for rank, index in enumerate(indices, start=1):
            keyword_rows.append({
                "topic_id": topic_id,
                "rank": rank,
                "keyword": str(feature_names[index]),
                "weight": float(weights[index]),
            })

    topic_keywords = pd.DataFrame(keyword_rows)

    topic_counts = document_topics["topic_id"].value_counts()
    summary_rows = []

    for topic_id in range(k):
        keywords = topic_keywords.loc[
            topic_keywords["topic_id"] == topic_id,
            "keyword",
        ].tolist()

        count = int(topic_counts.get(topic_id, 0))

        summary_rows.append({
            "topic_id": topic_id,
            "topic_name": f"Topic {topic_id}",
            "document_count": count,
            "percentage": count / len(document_topics),
            "top_keywords": json.dumps(
                keywords, ensure_ascii=False
            ),
        })

    topic_summary = pd.DataFrame(summary_rows)

    # Load the LDA Final runtime benchmark when available.
    # Resolve relative to the repository, not the current directory.
    project_root = Path(__file__).resolve().parents[2]
    runtime_path = (
        project_root
        / "results"
        / "lda"
        / "runtime"
        / "runtime_summary.csv"
    )

    runtime_seconds = None

    if runtime_path.is_file():
        runtime_df = pd.read_csv(runtime_path)

        required_columns = {
            "run",
            "fit_split",
            "inference_split",
            "stage",
            "repeats",
            "mean_seconds",
        }

        missing_columns = required_columns - set(runtime_df.columns)
        if missing_columns:
            raise ValueError(
                f"Missing LDA runtime columns: {sorted(missing_columns)}"
            )

        runtime_rows = runtime_df.loc[
            (runtime_df["run"] == "lda_k15_seed42_final_benchmark")
            & (runtime_df["fit_split"] == "train+val")
            & (runtime_df["inference_split"] == "test")
            & (runtime_df["stage"] == "runtime_seconds")
            & (runtime_df["repeats"] == 5)
        ]

        if len(runtime_rows) != 1:
            raise ValueError(
                "Expected exactly one LDA Final runtime benchmark row"
            )

        runtime_seconds = float(
            runtime_rows.iloc[0]["mean_seconds"]
        )

        if not np.isfinite(runtime_seconds) or runtime_seconds <= 0:
            raise ValueError("Invalid LDA Final runtime")

    # Runtime breakdown from reproducible LDA benchmarks.
    runtime_details = {
        "representation_seconds": None,
        "fit_seconds": None,
        "validation_inference_seconds": None,
        "test_inference_seconds": None,
        "evaluation_seconds": None,
        "runtime_scope": (
            "Final Train+Val -> Test: BoW representation, "
            "LDA fit, Test inference and evaluation; "
            "excludes data loading and artifact export. "
            "Validation inference is reported separately "
            "from the Development Train -> Validation benchmark."
        ),
        "runtime_mean": None,
        "runtime_std": None,
    }

    if runtime_path.is_file():
        def get_runtime(run_name, stage, field):
            matches = runtime_df.loc[
                (runtime_df["run"] == run_name)
                & (runtime_df["stage"] == stage)
                & (runtime_df["repeats"] == 5)
            ]

            if len(matches) != 1:
                raise ValueError(
                    f"Expected one benchmark row: {run_name}, {stage}"
                )

            value = float(matches.iloc[0][field])

            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    f"Invalid benchmark value: {run_name}, {stage}"
                )

            return value

        development_run = "lda_k15_seed42_development_benchmark"
        final_run = "lda_k15_seed42_final_benchmark"

        runtime_details.update({
            "representation_seconds": get_runtime(
                final_run, "representation_seconds", "mean_seconds"
            ),
            "fit_seconds": get_runtime(
                final_run, "fit_seconds", "mean_seconds"
            ),
            "validation_inference_seconds": get_runtime(
                development_run, "predict_seconds", "mean_seconds"
            ),
            "test_inference_seconds": get_runtime(
                final_run, "predict_seconds", "mean_seconds"
            ),
            "evaluation_seconds": get_runtime(
                final_run, "evaluation_seconds", "mean_seconds"
            ),
            "runtime_mean": get_runtime(
                final_run, "runtime_seconds", "mean_seconds"
            ),
            "runtime_std": get_runtime(
                final_run, "runtime_seconds", "std_seconds"
            ),
        })


    # Read supplementary Development Validation metrics.
    validation_metrics_path = (
        project_root
        / "results"
        / "lda"
        / "validation_supplement"
        / "validation_metrics.json"
    )

    validation_metrics = _read_json(validation_metrics_path)

    if (
        validation_metrics["fit_split"] != "train"
        or validation_metrics["evaluation_split"] != "val"
        or int(validation_metrics["n_topics"]) != 15
        or int(validation_metrics["random_state"]) != 42
        or validation_metrics["npmi_definition"]
        != "document_binary_common"
    ):
        raise ValueError("Invalid LDA Validation evaluation protocol")

    # Read M09 stability using reference seed 42.
    stability_path = (
        project_root
        / "results"
        / "lda"
        / "m09"
        / "stability_metrics.csv"
    )

    stability_df = pd.read_csv(stability_path)

    required_stability_columns = {
        "reference_seed",
        "candidate_seed",
        "k",
        "mean_matched_jaccard",
        "mean_document_cosine",
    }

    missing = required_stability_columns - set(stability_df.columns)
    if missing:
        raise ValueError(
            f"Missing LDA stability columns: {sorted(missing)}"
        )

    stability_rows = stability_df.loc[
        (stability_df["reference_seed"] == 42)
        & (stability_df["candidate_seed"] != 42)
        & (stability_df["k"] == 15)
    ].copy()

    if (
        len(stability_rows) != 4
        or stability_rows["candidate_seed"].nunique() != 4
    ):
        raise ValueError("Expected 4 M09 candidate seeds")

    stability_mean = float(
        stability_rows["mean_matched_jaccard"].mean()
    )
    stability_std = float(
        stability_rows["mean_matched_jaccard"].std(ddof=1)
    )

    document_cosine_mean = float(
        stability_rows["mean_document_cosine"].mean()
    )
    document_cosine_std = float(
        stability_rows["mean_document_cosine"].std(ddof=1)
    )

    metric_details = {
        "validation_ari": float(
            validation_metrics["validation_ari"]
        ),
        "validation_nmi": float(
            validation_metrics["validation_nmi"]
        ),
        "validation_npmi": float(
            validation_metrics["validation_npmi"]
        ),
        "test_ari": float(final_metrics["adjusted_rand_index"]),
        "test_nmi": float(final_metrics["normalized_mutual_info"]),
        "test_npmi": float(final_metrics["npmi_coherence"]),
        "perplexity": float(
            validation_metrics["validation_perplexity"]
        ),
        "topic_diversity": float(
            final_metrics["topic_diversity"]
        ),
        "stability_mean": stability_mean,
        "stability_std": stability_std,
        "document_cosine_mean": document_cosine_mean,
        "document_cosine_std": document_cosine_std,
    }

    for key, value in metric_details.items():
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"Invalid LDA metric: {key}")

    run_id = "lda_k15_seed42"
    model_name = "LDA K15"

    model_summary = {
        "model_name": model_name,
        "run_id": run_id,
        "algorithm": "LDA",
        "representation": "Bag-of-Words",
        "topic_count": k,
        "evaluation_protocol": {
            "validation": {
                "fit_split": "train",
                "evaluation_split": "val",
                "model": "LDA Development K15 seed42",
                "npmi_definition": "document_binary_common",
                "perplexity_scope": "Development Train -> Validation",
            },
            "test": {
                "fit_split": "train+val",
                "evaluation_split": "test",
                "model": "LDA Final K15 seed42",
                "npmi_definition": "document_binary_common",
            },
            "stability": {
                "metric": "matched_jaccard",
                "reference_seed": 42,
                "candidate_seeds": [52, 62, 72, 82],
                "std_definition": "sample_std_ddof_1",
            },
        },
        "outlier_rate": 0.0,
        "runtime_seconds": runtime_seconds,
        **runtime_details,
        "multi_topic": {
            "definition": "second_topic_probability >= threshold",
            "threshold": 0.20,
            "total_documents": len(document_topics),
            "multi_topic_count": int(
                document_topics["is_multi_topic"].sum()
            ),
            "multi_topic_rate": float(
                document_topics["is_multi_topic"].mean()
            ),
        },
        "metrics": {
            **metric_details,
            "npmi_coherence": float(
                final_metrics["npmi_coherence"]
            ),
            "npmi_coherence": float(
                final_metrics["npmi_coherence"]
            ),
            "adjusted_rand_index": float(
                final_metrics["adjusted_rand_index"]
            ),
            "normalized_mutual_info": float(
                final_metrics["normalized_mutual_info"]
            ),
            "document_count": len(document_topics),
            "train_count": int(
                (document_topics["split"] == "train").sum()
            ),
            "validation_count": int(
                (document_topics["split"] == "val").sum()
            ),
            "test_count": int(
                (document_topics["split"] == "test").sum()
            ),
        },

        "config": final_config,
    }

    manifest = {
        "schema_version": "1.0",
        "model_name": model_name,
        "run_id": run_id,
        "model_summary_path": "model_summary.json",
        "topic_summary_path": "topic_summary.csv",
        "document_topics_path": "document_topics.csv",
        "topic_keywords_path": "topic_keywords.csv",
        "config_path": "config_used.yaml",
    }

    if document_topics["article_id"].nunique() != 9274:
        raise ValueError("Incomplete document assignment")

    if not document_topics["topic_id"].between(0, 14).all():
        raise ValueError("Invalid topic ID")

    if len(topic_summary) != k or len(topic_keywords) != k * 10:
        raise ValueError("Topic summary/keywords mismatch")

    if topic_summary["document_count"].sum() != 9274:
        raise ValueError("Topic counts mismatch")


    # Load representatives selected from Train + Validation.
    representatives = pd.read_csv(
        representatives_path,
        encoding="utf-8-sig",
    )

    required_rep_columns = {
        "topic_id",
        "rank",
        "article_id",
        "title",
        "topic_probability",
    }

    if not required_rep_columns.issubset(representatives.columns):
        raise ValueError("Invalid representative columns")

    if len(representatives) != k * 3:
        raise ValueError("Expected 3 representatives per topic")

    if representatives["article_id"].duplicated().any():
        raise ValueError("Duplicate representative article IDs")

    for column in ("topic_id", "rank", "topic_probability"):
        representatives[column] = pd.to_numeric(
            representatives[column],
            errors="raise",
        )

    expected_pairs = {
        (topic_id, rank)
        for topic_id in range(k)
        for rank in (1, 2, 3)
    }

    actual_pairs = set(
        zip(
            representatives["topic_id"],
            representatives["rank"],
        )
    )

    if actual_pairs != expected_pairs:
        raise ValueError("Incomplete representative topic/rank pairs")

    if not representatives["topic_probability"].between(0, 1).all():
        raise ValueError("Invalid representative probabilities")

    representative_splits = representatives[
        ["article_id"]
    ].merge(
        document_topics[["article_id", "split"]],
        on="article_id",
        how="left",
        validate="one_to_one",
    )

    if not representative_splits["split"].isin(
        ["train", "val"]
    ).all():
        raise ValueError(
            "Representatives must belong to Train/Validation"
        )

    representatives["split"] = (
        representative_splits["split"].to_numpy()
    )

    output_dir.mkdir(parents=True, exist_ok=False)

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

    representatives.to_csv(
        output_dir / "representative_docs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    projection_2d.to_csv(
        output_dir / "projection_2d.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (output_dir / "config_used.yaml").open(
        "w", encoding="utf-8"
    ) as file:
        yaml.safe_dump(
            final_config,
            file,
            allow_unicode=True,
            sort_keys=False,
        )

    _write_json(
        output_dir / "model_summary.json",
        model_summary,
    )
    _write_json(
        output_dir / "manifest.json",
        manifest,
    )

    return load_artifact_bundle(output_dir)


def export_nmf_artifacts(
    source_dir: str | Path = "artifacts/nmf/nmf_p3_k10_seed42_v1",
    output_dir: str | Path = "artifacts/common/nmf/nmf_p3_k10_seed42_v1",
    result_dir: str | Path | None = None,
) -> ArtifactBundle:
    """Xuất Parquet NMF sang common artifact schema mà không sửa kết quả gốc."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    result_dir = (
        Path(result_dir)
        if result_dir is not None
        else Path("results/nmf") / source_dir.name
    )
    _require_files(
        [
            source_dir / "config.json",
            source_dir / "run_metadata.json",
            source_dir / "topic_keywords.json",
            source_dir / "representative_docs.json",
            result_dir / "test_metrics.json",
        ],
        model_label="NMF",
    )
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")

    config = _read_json(source_dir / "config.json")
    metadata = _read_json(source_dir / "run_metadata.json")
    metrics = _read_json(result_dir / "test_metrics.json")
    assignments = read_nmf_parquet_assignments(source_dir)
    threshold = float(config.get("analysis", {}).get("secondary_topic_min_share", 0.20))
    gap_threshold = float(config.get("analysis", {}).get("top_two_gap_filter", 0.10))
    assignments["is_multi_topic"] = assignments["second_topic_share"].ge(threshold)
    assignments["is_close_top_two"] = assignments["top_two_share_gap"].le(gap_threshold)
    topic_count = int(metadata.get("selected_n_topics", 0))
    run_id = str(metadata.get("run_id", source_dir.name))
    if topic_count < 2 or run_id != source_dir.name:
        raise ValueError("NMF metadata phải có topic count và run_id khớp thư mục")
    if set(assignments["split"].astype(str)) != {"train", "val", "test"}:
        raise ValueError("NMF common artifacts cần đủ train, val và test")
    expected_split_counts = metadata.get("split_counts")
    if expected_split_counts:
        actual_split_counts = {
            split: int(count)
            for split, count in assignments["split"].value_counts().items()
        }
        if actual_split_counts != expected_split_counts:
            raise ValueError("Số bài theo split không khớp NMF run metadata")

    all_topic_ids = set(assignments["topic_id"].astype(int))
    valid_topic_ids = {
        topic_id
        for topic_id in all_topic_ids
        if topic_id == -1 or 0 <= topic_id < topic_count
    }
    if valid_topic_ids != all_topic_ids:
        raise ValueError("topic_id NMF nằm ngoài số topic đã khai báo")
    regular_topic_ids = list(range(topic_count))

    raw_keywords = _read_json(source_dir / "topic_keywords.json")
    if not isinstance(raw_keywords, list):
        raise ValueError("topic_keywords.json của NMF phải là danh sách")
    keyword_rows: list[dict[str, Any]] = []
    keywords_by_topic: dict[int, list[str]] = {}
    for topic in raw_keywords:
        topic_id = normalize_nmf_topic_id(topic.get("topic_id"))
        if topic_id == -1 or topic_id not in regular_topic_ids:
            raise ValueError("topic_keywords.json chứa topic_id không hợp lệ")
        if topic_id in keywords_by_topic:
            raise ValueError(f"topic_keywords.json lặp topic_id {topic_id}")
        terms = topic.get("keywords")
        if not isinstance(terms, list) or not terms:
            raise ValueError(f"Topic {topic_id} thiếu keyword")
        keywords_by_topic[topic_id] = []
        for rank, item in enumerate(terms, start=1):
            keyword = str(item.get("term", "")).strip()
            weight = float(item.get("weight"))
            if not keyword or not np.isfinite(weight) or weight < 0:
                raise ValueError(f"Keyword không hợp lệ trong topic {topic_id}")
            keywords_by_topic[topic_id].append(keyword)
            keyword_rows.append(
                {
                    "topic_id": topic_id,
                    "rank": rank,
                    "keyword": keyword,
                    "weight": weight,
                }
            )
    if set(keywords_by_topic) != set(regular_topic_ids):
        raise ValueError("topic_keywords.json phải có đúng một danh sách cho mỗi topic")
    topic_keywords = pd.DataFrame(keyword_rows)

    display_mapping_path = source_dir / "topic_display_mapping.json"
    labels: dict[int, str] = {}
    mapping_is_reviewed = False
    if display_mapping_path.is_file():
        display_mapping = _read_json(display_mapping_path)
        display_topics = display_mapping.get("topics", [])
        mapping_is_reviewed = display_mapping.get("review_status") == "approved"
        for topic in display_topics:
            topic_id = normalize_nmf_topic_id(topic.get("topic_id"))
            label = str(topic.get("display_label", "")).strip()
            if topic_id in regular_topic_ids and label:
                labels[topic_id] = label

    total_documents = len(assignments)
    counts = assignments["topic_id"].value_counts()
    summary_rows = []
    for topic_id in regular_topic_ids:
        display_name = labels.get(topic_id)
        if display_name and not mapping_is_reviewed:
            topic_name = f"NMF_{topic_id:02d} — {display_name} (nhãn tạm)"
        else:
            topic_name = display_name or f"Topic {topic_id}"
        summary_rows.append(
            {
                "topic_id": topic_id,
                "topic_name": topic_name,
                "document_count": int(counts.get(topic_id, 0)),
                "percentage": float(counts.get(topic_id, 0) / total_documents),
                "top_keywords": ", ".join(keywords_by_topic[topic_id]),
            }
        )
    if -1 in all_topic_ids:
        outlier_count = int(counts.get(-1, 0))
        summary_rows.append(
            {
                "topic_id": -1,
                "topic_name": "Outlier / Unassigned",
                "document_count": outlier_count,
                "percentage": outlier_count / total_documents,
                "top_keywords": "",
            }
        )
    topic_summary = pd.DataFrame(summary_rows)
    if int(topic_summary["document_count"].sum()) != total_documents:
        raise ValueError("Topic summary không bao phủ đủ assignments")

    representative_rows = _read_json(source_dir / "representative_docs.json")
    if not isinstance(representative_rows, list):
        raise ValueError("representative_docs.json của NMF phải là danh sách")
    representatives = pd.DataFrame(representative_rows)
    required_rep_columns = {"topic_id", "rank", "article_id", "title", "topic_share"}
    if not required_rep_columns.issubset(representatives.columns):
        raise ValueError("representative_docs.json thiếu trường bắt buộc")
    if representatives.empty:
        raise ValueError("NMF chưa có bài đại diện")
    representatives["topic_id"] = representatives["topic_id"].map(
        normalize_nmf_topic_id
    )
    representatives["article_id"] = representatives["article_id"].astype(str)
    if not representatives["topic_id"].isin(regular_topic_ids).all():
        raise ValueError("Bài đại diện có topic_id không hợp lệ")
    if not representatives["article_id"].isin(assignments["article_id"]).all():
        raise ValueError("Bài đại diện không có trong document assignments")
    representatives["topic_share"] = pd.to_numeric(
        representatives["topic_share"], errors="raise"
    )
    if (
        not np.isfinite(representatives["topic_share"]).all()
        or not representatives["topic_share"].between(0, 1).all()
    ):
        raise ValueError("topic_share của bài đại diện phải nằm trong [0, 1]")
    representative_splits = assignments[
        ["article_id", "split", "source", "published_at"]
    ].drop_duplicates("article_id")
    representatives = representatives.merge(
        representative_splits,
        on="article_id",
        how="left",
        validate="many_to_one",
    )
    representatives = representatives.rename(
        columns={"topic_share": "primary_topic_share"}
    )
    representatives["topic_probability"] = pd.NA
    representatives["display_label"] = representatives["topic_id"].map(labels)

    share_columns = [
        column
        for column in assignments.columns
        if re.fullmatch(r"topic_\d+_share", str(column))
    ]
    share_columns = sorted(share_columns, key=lambda name: int(name.split("_")[1]))
    share_topic_ids = {int(name.split("_")[1]) for name in share_columns}
    if share_topic_ids != set(regular_topic_ids):
        raise ValueError("Số cột topic share không khớp topic count")
    feature_values = assignments[share_columns].to_numpy(dtype=np.float64)
    development_mask = assignments["split"].isin(["train", "val"]).to_numpy()
    if int(development_mask.sum()) < 2:
        raise ValueError("Cần ít nhất hai bài development để tạo PCA projection")
    projection = PCA(n_components=2, svd_solver="full")
    projection.fit(feature_values[development_mask])
    coordinates = projection.transform(feature_values)
    projection_2d = pd.DataFrame(
        {
            "article_id": assignments["article_id"],
            "split": assignments["split"],
            "topic_id": assignments["topic_id"],
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
        }
    )

    output_dir.mkdir(parents=True)
    topic_summary.to_csv(
        output_dir / "topic_summary.csv", index=False, encoding="utf-8-sig"
    )
    assignments.to_csv(
        output_dir / "document_topics.csv", index=False, encoding="utf-8-sig"
    )
    topic_keywords.to_csv(
        output_dir / "topic_keywords.csv", index=False, encoding="utf-8-sig"
    )
    representatives.to_csv(
        output_dir / "representative_docs.csv", index=False, encoding="utf-8-sig"
    )
    projection_2d.to_csv(
        output_dir / "projection_2d.csv", index=False, encoding="utf-8-sig"
    )
    with (output_dir / "config_used.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)

    model_name = f"NMF {run_id}"
    numeric_metrics = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(value)
    }
    numeric_metrics["train_count"] = int((assignments["split"] == "train").sum())
    numeric_metrics["validation_count"] = int(
        (assignments["split"] == "val").sum()
    )
    numeric_metrics["test_count"] = int((assignments["split"] == "test").sum())
    dashboard_metrics_path = result_dir / "dashboard_metrics.json"
    if dashboard_metrics_path.is_file():
        numeric_metrics.update({
            key: value for key, value in _read_json(dashboard_metrics_path).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            and np.isfinite(value)
        })
    model_summary = {
        "model_name": model_name,
        "run_id": run_id,
        "algorithm": "NMF",
        "representation": "TF-IDF",
        "topic_count": topic_count,
        "outlier_rate": float(assignments["is_outlier"].mean()),
        "runtime_seconds": metadata.get("runtime_seconds"),
        "runtime_scope": metadata.get("runtime_scope"),
        "evaluation_split": "test",
        "representative_split": metadata.get("representative_split", "source_artifact"),
        "metrics": numeric_metrics,
        "config": config,
    }
    manifest = {
        "schema_version": "1.0",
        "model_name": model_name,
        "run_id": run_id,
        "model_summary_path": "model_summary.json",
        "topic_summary_path": "topic_summary.csv",
        "document_topics_path": "document_topics.csv",
        "topic_keywords_path": "topic_keywords.csv",
        "config_path": "config_used.yaml",
    }
    _write_json(output_dir / "model_summary.json", model_summary)
    _write_json(output_dir / "manifest.json", manifest)
    return load_artifact_bundle(output_dir)
