"""Chạy thí nghiệm M04-M06 theo Dataset V1 và split đã khóa."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from experiments.nmf.metrics import (
    normalized_entropy,
    normalized_reconstruction_error,
    npmi_coherence,
    topic_diversity,
)
from src.models.nmf import NMFConfig, NMFTopicModel


LOGGER = logging.getLogger("nmf_experiment")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALID_SPLITS = {"train", "val", "test"}


def resolve_project_path(value: str | Path) -> Path:
    """Chuyển đường dẫn tương đối trong config thành đường dẫn từ project root."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    """Tính SHA-256 theo từng khối để không nạp toàn bộ file lớn vào RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    """Đọc YAML và yêu cầu cấu trúc cấp cao nhất là mapping."""
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy config: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config NMF phải là một YAML mapping")
    return config


def load_dataset_v1(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Đọc Dataset V1, xác minh manifest và gắn split theo article_id."""
    data_config = config["data"]
    dataset_path = resolve_project_path(data_config["dataset_path"])
    manifest_path = resolve_project_path(data_config["dataset_manifest_path"])
    split_path = resolve_project_path(data_config["split_manifest_path"])
    for path in (dataset_path, manifest_path, split_path):
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy đầu vào bắt buộc: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    variant_name = data_config["preprocessing_variant"]
    variant_info = manifest.get("variants", {}).get(variant_name)
    if manifest.get("review_status") != "approved" or not variant_info:
        raise ValueError("Dataset manifest chưa duyệt hoặc thiếu preprocessing variant")
    if manifest.get("dataset_version") != data_config["dataset_version"]:
        raise ValueError("dataset_version trong config không khớp manifest")
    if sha256_file(dataset_path) != variant_info.get("sha256"):
        raise ValueError("SHA-256 của dataset không khớp manifest đã duyệt")

    id_column = data_config["id_column"]
    text_column = data_config["text_column"]
    dataset = pd.read_csv(
        dataset_path,
        encoding=data_config.get("encoding", "utf-8-sig"),
        dtype={id_column: "string"},
        low_memory=False,
    )
    required_columns = {id_column, text_column, "title", "published_at", "source"}
    missing_columns = required_columns - set(dataset.columns)
    if missing_columns:
        raise ValueError(f"Dataset thiếu cột: {sorted(missing_columns)}")
    if dataset[id_column].isna().any() or dataset[id_column].duplicated().any():
        raise ValueError("article_id phải đầy đủ và duy nhất trong Dataset V1")
    if dataset[text_column].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError(f"Cột {text_column} có văn bản rỗng")

    split_manifest = pd.read_csv(split_path, encoding="utf-8-sig", dtype={id_column: "string"})
    if not {id_column, "split"}.issubset(split_manifest.columns):
        raise ValueError("Split manifest thiếu article_id hoặc split")
    if split_manifest[id_column].duplicated().any():
        raise ValueError("Split manifest có article_id trùng")
    if not set(split_manifest["split"].unique()).issubset(VALID_SPLITS):
        raise ValueError("Split manifest chứa tên split không hợp lệ")

    # Merge one-to-one giúp phát hiện cả ID thiếu lẫn ID thừa giữa hai nguồn.
    merged = dataset.merge(
        split_manifest[[id_column, "split"]],
        on=id_column,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Dataset V1 và split manifest không có cùng tập article_id")
    return merged.drop(columns="_merge"), manifest


def build_vectorizer(config: dict[str, Any]) -> TfidfVectorizer:
    """Tạo TF-IDF đúng tham số đã lưu trong config."""
    representation = config["representation"]
    return TfidfVectorizer(
        ngram_range=tuple(representation["ngram_range"]),
        min_df=representation["min_df"],
        max_df=representation["max_df"],
        max_features=representation["max_features"],
        sublinear_tf=representation["sublinear_tf"],
        token_pattern=representation["token_pattern"],
        dtype=np.float64,
    )


def build_model(config: dict[str, Any], n_topics: int) -> NMFTopicModel:
    """Tạo NMF từ config để mọi run dùng cùng quy tắc."""
    model = config["model"]
    return NMFTopicModel(
        NMFConfig(
            n_topics=n_topics,
            random_seed=int(model["random_seed"]),
            init=model["init"],
            solver=model["solver"],
            beta_loss=model["beta_loss"],
            max_iter=int(model["max_iter"]),
            tolerance=float(model["tolerance"]),
        )
    )


def topic_feature_ids(
    model: NMFTopicModel,
    feature_names: np.ndarray,
    top_n: int,
) -> tuple[list[list[str]], list[list[int]]]:
    """Ánh xạ top words sang chỉ số cột để tính coherence."""
    feature_to_id = {term: index for index, term in enumerate(feature_names)}
    terms_by_topic: list[list[str]] = []
    ids_by_topic: list[list[int]] = []
    for topic_index in range(model.config.n_topics):
        terms = [term for term, _ in model.top_terms(topic_index, top_n)]
        terms_by_topic.append(terms)
        ids_by_topic.append([feature_to_id[term] for term in terms])
    return terms_by_topic, ids_by_topic


def evaluate_model(
    model: NMFTopicModel,
    matrix,
    feature_names: np.ndarray,
    top_n: int,
) -> tuple[dict[str, float | int | bool], list[float]]:
    """Tính metric trên validation hoặc test mà không fit lại trên split đó."""
    distributions = model.transform(matrix)
    # Khôi phục scale W gốc từ phân phối chuẩn hóa chỉ dùng cho reconstruction.
    raw_weights = model.estimator_.transform(matrix)
    terms_by_topic, ids_by_topic = topic_feature_ids(model, feature_names, top_n)
    coherence, per_topic = npmi_coherence(matrix, ids_by_topic)
    metrics: dict[str, float | int | bool] = {
        "n_topics": model.config.n_topics,
        "npmi_coherence": coherence,
        "topic_diversity": topic_diversity(terms_by_topic),
        "normalized_reconstruction_error": normalized_reconstruction_error(
            matrix,
            raw_weights,
            model.topic_word_weights_,
        ),
        "mean_primary_topic_share": float(distributions.max(axis=1).mean()),
        "n_iter": int(model.estimator_.n_iter_),
        "converged": int(model.estimator_.n_iter_) < model.config.max_iter,
    }
    return metrics, per_topic


def choose_topic_count(metrics: pd.DataFrame) -> int:
    """Đề xuất K từ validation, không đọc bất kỳ metric nào của test."""
    ranked = metrics.sort_values(
        [
            "npmi_coherence",
            "topic_diversity",
            "normalized_reconstruction_error",
            "n_topics",
        ],
        ascending=[False, False, True, True],
        kind="stable",
    )
    return int(ranked.iloc[0]["n_topics"])


def create_run_directories(
    config: dict[str, Any],
    stage: str,
    run_id: str | None,
) -> tuple[str, Path, Path]:
    """Tạo thư mục artifact/result mới và ngăn ghi đè run cũ."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    selected_run_id = run_id or f"nmf_p3_{stage}_seed42_{timestamp}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", selected_run_id):
        raise ValueError("run_id chỉ được chứa chữ, số, dấu chấm, gạch dưới và gạch nối")

    artifact_dir = resolve_project_path(config["outputs"]["artifact_root"]) / selected_run_id
    result_dir = resolve_project_path(config["outputs"]["result_root"]) / selected_run_id
    if artifact_dir.exists() or result_dir.exists():
        raise FileExistsError(f"Run đã tồn tại: {selected_run_id}")
    artifact_dir.mkdir(parents=True)
    result_dir.mkdir(parents=True)
    return selected_run_id, artifact_dir, result_dir


def save_selection_plot(metrics: pd.DataFrame, output_path: Path) -> None:
    """Vẽ metric theo K để phục vụ M05, paper và demo contribution."""
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    columns = [
        ("npmi_coherence", "NPMI coherence ↑"),
        ("topic_diversity", "Topic diversity ↑"),
        ("normalized_reconstruction_error", "Reconstruction error ↓"),
    ]
    for axis, (column, title) in zip(axes, columns):
        axis.plot(metrics["n_topics"], metrics[column], marker="o")
        axis.set_title(title)
        axis.set_xlabel("Số topic K")
        axis.grid(alpha=0.25)
    figure.suptitle("NMF: lựa chọn số topic trên validation")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_selection(
    dataset: pd.DataFrame,
    config: dict[str, Any],
    run_id: str | None,
) -> None:
    """Fit trên train và chọn K bằng validation cho task M05."""
    text_column = config["data"]["text_column"]
    train = dataset.loc[dataset["split"] == "train"]
    validation = dataset.loc[dataset["split"] == "val"]
    vectorizer = build_vectorizer(config)
    train_matrix = vectorizer.fit_transform(train[text_column].astype(str))
    validation_matrix = vectorizer.transform(validation[text_column].astype(str))
    feature_names = vectorizer.get_feature_names_out()
    top_n = int(config["topic_selection"]["coherence_top_n"])

    records: list[dict[str, Any]] = []
    for n_topics in config["topic_selection"]["candidates"]:
        LOGGER.info("Đang fit NMF K=%s trên train", n_topics)
        model = build_model(config, int(n_topics)).fit(train_matrix, feature_names)
        metrics, _ = evaluate_model(model, validation_matrix, feature_names, top_n)
        metrics["fit_split"] = "train"
        metrics["evaluation_split"] = "val"
        records.append(metrics)

    metrics_frame = pd.DataFrame(records).sort_values("n_topics")
    recommended_k = choose_topic_count(metrics_frame)
    selected_run_id, artifact_dir, result_dir = create_run_directories(
        config, "select", run_id
    )
    metrics_frame.to_csv(
        result_dir / "topic_count_metrics.csv", index=False, encoding="utf-8-sig"
    )
    save_selection_plot(metrics_frame, result_dir / "topic_count_selection.png")
    metadata = base_metadata(selected_run_id, "select", dataset, config)
    metadata.update(
        {
            "recommended_n_topics": recommended_k,
            "selection_split": "val",
            "test_accessed": False,
            "selection_rule": (
                "NPMI cao nhất; hòa điểm ưu tiên diversity cao, "
                "reconstruction error thấp rồi K nhỏ"
            ),
        }
    )
    write_json(artifact_dir / "config.json", config)
    write_json(artifact_dir / "run_metadata.json", metadata)
    LOGGER.info("K được đề xuất từ validation: %s", recommended_k)
    LOGGER.info("Kết quả M05: %s", result_dir)


def build_document_outputs(
    test: pd.DataFrame,
    distributions: np.ndarray,
    model: NMFTopicModel,
    config: dict[str, Any],
    per_topic_coherence: list[float],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    """Tạo assignments, keywords, representative docs và case đa topic."""
    id_column = config["data"]["id_column"]
    gold_column = config["data"]["gold_column"]
    analysis = config["analysis"]
    threshold = float(analysis["secondary_topic_min_share"])
    max_topics = int(analysis["max_topics_per_document"])
    entropies = normalized_entropy(distributions)
    order = np.argsort(-distributions, axis=1, kind="stable")

    document_topics = test[
        [id_column, "title", "published_at", "source", gold_column, "split"]
    ].reset_index(drop=True).copy()
    primary_indexes = order[:, 0]
    document_topics["primary_topic_id"] = [f"NMF_{value:02d}" for value in primary_indexes]
    document_topics["primary_topic_share"] = distributions[np.arange(len(test)), primary_indexes]
    document_topics["secondary_topic_id"] = [f"NMF_{value:02d}" for value in order[:, 1]]
    document_topics["secondary_topic_share"] = distributions[np.arange(len(test)), order[:, 1]]
    document_topics["top_two_share_gap"] = (
        document_topics["primary_topic_share"] - document_topics["secondary_topic_share"]
    )
    document_topics["normalized_topic_entropy"] = entropies

    assigned_topics: list[str] = []
    for row, ranked_indexes in zip(distributions, order):
        selected = [int(ranked_indexes[0])]
        for topic_index in ranked_indexes[1:]:
            if len(selected) >= max_topics:
                break
            if row[topic_index] >= threshold:
                selected.append(int(topic_index))
        assigned_topics.append(
            json.dumps([f"NMF_{value:02d}" for value in selected], ensure_ascii=False)
        )
    document_topics["assigned_topic_ids"] = assigned_topics
    document_topics["is_multi_topic"] = [len(json.loads(value)) > 1 for value in assigned_topics]
    for topic_index in range(model.config.n_topics):
        document_topics[f"topic_{topic_index:02d}_share"] = distributions[:, topic_index]

    keyword_records: list[dict[str, Any]] = []
    representative_records: list[dict[str, Any]] = []
    representative_count = int(analysis["representative_documents_per_topic"])
    excerpt_size = int(analysis["preview_characters"])
    test_reset = test.reset_index(drop=True)
    for topic_index in range(model.config.n_topics):
        topic_id = f"NMF_{topic_index:02d}"
        keyword_records.append(
            {
                "topic_id": topic_id,
                "keywords": [
                    {"term": term, "weight": weight}
                    for term, weight in model.top_terms(topic_index, int(analysis["top_words"]))
                ],
                "npmi_coherence": per_topic_coherence[topic_index],
                "manual_label": "",
                "interpretability_rating_1_to_5": None,
                "review_notes": "",
            }
        )
        representative_indexes = np.argsort(
            -distributions[:, topic_index], kind="stable"
        )[:representative_count]
        for rank, document_index in enumerate(representative_indexes, start=1):
            article = test_reset.iloc[document_index]
            representative_records.append(
                {
                    "topic_id": topic_id,
                    "rank": rank,
                    "article_id": article[id_column],
                    "topic_share": float(distributions[document_index, topic_index]),
                    "title": article["title"],
                    "category_gold_diagnostic": article.get(gold_column, ""),
                    "text_excerpt": str(article[config["data"]["text_column"]])[:excerpt_size],
                }
            )

    # Case có gap nhỏ và entropy cao thể hiện ranh giới topic mơ hồ rõ nhất.
    multi_topic_cases = document_topics.loc[document_topics["is_multi_topic"]].copy()
    multi_topic_cases = multi_topic_cases.sort_values(
        ["top_two_share_gap", "normalized_topic_entropy"],
        ascending=[True, False],
    ).head(int(analysis["multi_topic_cases_limit"]))
    text_by_id = test_reset.set_index(id_column)[config["data"]["text_column"]]
    multi_topic_cases["text_excerpt"] = (
        multi_topic_cases[id_column]
        .map(text_by_id)
        .fillna("")
        .astype(str)
        .str.slice(0, excerpt_size)
    )
    multi_topic_cases["selection_reason"] = (
        f"secondary_topic_share >= {threshold:.2f}; "
        "ưu tiên gap nhỏ và entropy cao"
    )
    multi_topic_cases["manual_analysis_note"] = ""
    return document_topics, keyword_records, representative_records, multi_topic_cases


def run_finalization(
    dataset: pd.DataFrame,
    config: dict[str, Any],
    n_topics_override: int | None,
    run_id: str | None,
) -> None:
    """Refit train+val bằng K đã chốt và đánh giá test đúng một lần."""
    configured_k = config["topic_selection"].get("selected_n_topics")
    n_topics = n_topics_override if n_topics_override is not None else configured_k
    if n_topics is None:
        raise ValueError(
            "Stage finalize cần --n-topics hoặc topic_selection.selected_n_topics"
        )
    n_topics = int(n_topics)
    if n_topics not in {int(value) for value in config["topic_selection"]["candidates"]}:
        raise ValueError("K cuối phải thuộc danh sách candidates đã đánh giá")

    text_column = config["data"]["text_column"]
    development = dataset.loc[dataset["split"].isin(["train", "val"])]
    test = dataset.loc[dataset["split"] == "test"].reset_index(drop=True)
    vectorizer = build_vectorizer(config)
    development_matrix = vectorizer.fit_transform(development[text_column].astype(str))
    test_matrix = vectorizer.transform(test[text_column].astype(str))
    feature_names = vectorizer.get_feature_names_out()
    model = build_model(config, n_topics).fit(development_matrix, feature_names)
    metrics, per_topic = evaluate_model(
        model,
        test_matrix,
        feature_names,
        int(config["topic_selection"]["coherence_top_n"]),
    )
    distributions = model.transform(test_matrix)
    document_topics, keywords, representatives, multi_topic_cases = build_document_outputs(
        test, distributions, model, config, per_topic
    )

    gold_column = config["data"]["gold_column"]
    if gold_column in test.columns:
        metrics["adjusted_rand_index"] = float(
            adjusted_rand_score(test[gold_column], document_topics["primary_topic_id"])
        )
        metrics["normalized_mutual_info"] = float(
            normalized_mutual_info_score(test[gold_column], document_topics["primary_topic_id"])
        )

    selected_run_id, artifact_dir, result_dir = create_run_directories(
        config, "final", run_id
    )
    model_dir = artifact_dir / "model"
    model_dir.mkdir()
    joblib.dump(
        {"vectorizer": vectorizer, "topic_model": model},
        model_dir / "nmf_pipeline.joblib",
    )
    development_topics = development[[config["data"]["id_column"], "split"]].copy()
    development_distributions = model.training_topic_distributions()
    for topic_index in range(n_topics):
        development_topics[f"topic_{topic_index:02d}_share"] = (
            development_distributions[:, topic_index]
        )
    development_topics.to_parquet(
        artifact_dir / "development_document_topics.parquet",
        index=False,
    )
    document_topics.to_parquet(artifact_dir / "document_topics.parquet", index=False)
    pd.DataFrame(model.topic_word_weights_, columns=feature_names).to_parquet(
        artifact_dir / "topic_word_matrix.parquet", index=False
    )
    write_json(artifact_dir / "topic_keywords.json", keywords)
    write_json(artifact_dir / "representative_docs.json", representatives)
    # Lưu K hiệu lực kể cả khi người chạy truyền qua CLI để artifact tự mô tả đủ.
    effective_config = {
        **config,
        "topic_selection": {
            **config["topic_selection"],
            "selected_n_topics": n_topics,
        },
    }
    write_json(artifact_dir / "config.json", effective_config)

    multi_topic_cases.to_csv(
        result_dir / "multi_topic_cases.csv", index=False, encoding="utf-8-sig"
    )
    write_json(result_dir / "test_metrics.json", metrics)
    metadata = base_metadata(selected_run_id, "finalize", dataset, config)
    metadata.update(
        {
            "selected_n_topics": n_topics,
            "fit_split": "train+val",
            "evaluation_split": "test",
            "test_accessed": True,
            "model_file": "model/nmf_pipeline.joblib",
        }
    )
    write_json(artifact_dir / "run_metadata.json", metadata)
    LOGGER.info("Đã hoàn tất M04/M06 với K=%s", n_topics)
    LOGGER.info("Artifact chuẩn: %s", artifact_dir)
    LOGGER.info("Kết quả đánh giá: %s", result_dir)


def base_metadata(
    run_id: str,
    stage: str,
    dataset: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Tạo metadata chung để truy vết run và contribution evidence."""
    split_counts = dataset["split"].value_counts().to_dict()
    return {
        "run_id": run_id,
        "method": "NMF",
        "tasks": ["M04", "M05", "M06"],
        "stage": stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": config["data"]["dataset_version"],
        "preprocessing_variant": config["data"]["preprocessing_variant"],
        "dataset_sha256": sha256_file(resolve_project_path(config["data"]["dataset_path"])),
        "split_counts": {key: int(value) for key, value in split_counts.items()},
        "random_seed": int(config["model"]["random_seed"]),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }


def write_json(path: Path, value: Any) -> None:
    """Ghi JSON UTF-8 với kiểu số NumPy được chuyển về kiểu Python."""
    def convert(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Không thể serialize kiểu {type(item).__name__}")

    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=convert),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh cho hai stage an toàn với test set."""
    parser = argparse.ArgumentParser(description="NMF pipeline cho M04-M06")
    parser.add_argument("--config", default="configs/nmf.yaml")
    parser.add_argument("--stage", choices=["select", "finalize"], default="select")
    parser.add_argument("--n-topics", type=int, help="K đã chốt từ stage select")
    parser.add_argument("--run-id", help="Tên run theo convention của nhóm")
    return parser.parse_args()


def main() -> int:
    """Điều phối load dữ liệu và stage thí nghiệm được yêu cầu."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()
    config = load_yaml(resolve_project_path(args.config))
    dataset, manifest = load_dataset_v1(config)
    LOGGER.info(
        "Đã xác minh Dataset %s, quality gate %s",
        manifest["dataset_version"],
        manifest["quality_gate"],
    )
    if args.stage == "select":
        run_selection(dataset, config, args.run_id)
    else:
        run_finalization(dataset, config, args.n_topics, args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
