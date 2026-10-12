"""Chạy các thí nghiệm embedding và tham số BERTopic cho M14."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer

from src.data.dataset_loader import load_split_dataset
from src.evaluation.bertopic_artifacts import (
    build_document_topics,
    build_topic_keywords,
    build_topic_summary,
)
from src.evaluation.bertopic_metrics import (
    calculate_outlier_rate,
    calculate_topic_diversity,
)
from src.models.bertopic_pipeline import (
    build_bertopic_model,
    encode_documents,
    fit_bertopic_model,
    load_bertopic_config,
)


def parse_arguments() -> argparse.Namespace:
    """Đọc tham số chạy thí nghiệm M14."""
    parser = argparse.ArgumentParser(
        description="Chạy thí nghiệm BERTopic cho task M14."
    )
    parser.add_argument(
        "--config",
        default="configs/bertopic_experiments.yaml",
        help="Cấu hình danh sách thí nghiệm.",
    )
    parser.add_argument(
        "--base-config",
        default="configs/bertopic.yaml",
        help="Cấu hình BERTopic nền của M13.",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Mã thí nghiệm cần chạy.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số bài mỗi split để chạy thử.",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Đọc một file cấu hình YAML."""
    config_path = Path(path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy cấu hình: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Cấu hình phải có dạng dictionary.")

    return config


def select_run(
    experiment_config: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """Chọn cấu hình tương ứng với run_id."""
    for run in experiment_config["runs"]:
        if run["run_id"] == run_id:
            return run

    raise ValueError(f"Không tìm thấy run_id: {run_id}")


def build_run_config(
    base_config: dict[str, Any],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    """Ghép cấu hình M13 với tham số của một run M14."""
    config = copy.deepcopy(base_config)

    config["embedding"] = copy.deepcopy(
        run_config["embedding"]
    )
    config["umap"] = copy.deepcopy(run_config["umap"])
    config["hdbscan"] = copy.deepcopy(
        run_config["hdbscan"]
    )
    config["experiment"]["run_id"] = run_config["run_id"]

    return config


def prepare_documents(
    dataset: pd.DataFrame,
    text_column: str,
) -> list[str]:
    """Lấy danh sách văn bản và kiểm tra dữ liệu rỗng."""
    if text_column not in dataset.columns:
        raise ValueError(
            f"Dataset không có cột văn bản: {text_column}"
        )

    documents = (
        dataset[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .tolist()
    )

    if any(not document for document in documents):
        raise ValueError("Dataset có văn bản rỗng.")

    return documents


def update_summary(
    summary_path: Path,
    result: dict[str, Any],
) -> None:
    """Thêm hoặc cập nhật một run trong bảng tổng hợp."""
    current_result = pd.DataFrame([result])

    if summary_path.is_file():
        previous_results = pd.read_csv(summary_path)
        current_result = pd.concat(
            [previous_results, current_result],
            ignore_index=True,
        )
        current_result = current_result.drop_duplicates(
            subset=["run_id"],
            keep="last",
        )

    current_result.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    """Fit trên train và đánh giá run trên validation."""
    arguments = parse_arguments()

    experiment_config = load_yaml(arguments.config)
    base_config = load_bertopic_config(
        arguments.base_config
    )

    protocol = experiment_config["experiment"]

    if protocol["test_split_locked"] is not True:
        raise ValueError("M14 yêu cầu khóa tập test.")

    if (
        protocol["fit_split"] == "test"
        or protocol["evaluation_split"] == "test"
    ):
        raise ValueError("M14 không được sử dụng tập test.")

    selected_run = select_run(
        experiment_config,
        arguments.run_id,
    )
    run_config = build_run_config(
        base_config,
        selected_run,
    )

    dataset_config = base_config["dataset"]

    train_dataset = load_split_dataset(
        dataset_config["path"],
        dataset_config["split_manifest_path"],
        protocol["fit_split"],
    )
    validation_dataset = load_split_dataset(
        dataset_config["path"],
        dataset_config["split_manifest_path"],
        protocol["evaluation_split"],
    )

    if arguments.limit is not None:
        if arguments.limit <= 0:
            raise ValueError("--limit phải lớn hơn 0.")

        train_dataset = train_dataset.head(
            arguments.limit
        ).copy()
        validation_dataset = validation_dataset.head(
            arguments.limit
        ).copy()

    text_column = dataset_config["text_variant"]
    train_documents = prepare_documents(
        train_dataset,
        text_column,
    )
    validation_documents = prepare_documents(
        validation_dataset,
        text_column,
    )

    embedding_config = run_config["embedding"]
    model_arguments: dict[str, Any] = {}

    if embedding_config.get("device"):
        model_arguments["device"] = embedding_config["device"]

    print(f"Run: {arguments.run_id}")
    print(f"Train: {len(train_documents)} bài")
    print(f"Validation: {len(validation_documents)} bài")
    print(f"Embedding: {embedding_config['model_name']}")

    start_time = time.perf_counter()

    embedding_model = SentenceTransformer(
        embedding_config["model_name"],
        **model_arguments,
    )

    train_embeddings = encode_documents(
        embedding_model,
        train_documents,
        run_config,
    )
    validation_embeddings = encode_documents(
        embedding_model,
        validation_documents,
        run_config,
    )

    topic_model = build_bertopic_model(
        run_config,
        embedding_model=embedding_model,
    )

    train_topics, train_probabilities = fit_bertopic_model(
        topic_model,
        train_documents,
        train_embeddings,
    )

    validation_topics, validation_probabilities = (
        topic_model.transform(
            validation_documents,
            embeddings=validation_embeddings,
        )
    )

    runtime_seconds = time.perf_counter() - start_time

    topic_words = topic_model.get_topics()
    topic_count = len(
        {
            int(topic)
            for topic in train_topics
            if int(topic) != -1
        }
    )

    result = {
        "run_id": selected_run["run_id"],
        "description": selected_run["description"],
        "embedding_model": embedding_config["model_name"],
        "train_count": len(train_documents),
        "validation_count": len(validation_documents),
        "topic_count": topic_count,
        "train_outlier_rate": calculate_outlier_rate(
            train_topics
        ),
        "validation_outlier_rate": calculate_outlier_rate(
            validation_topics
        ),
        "topic_diversity": calculate_topic_diversity(
            topic_words,
            top_n=base_config["topic_representation"][
                "top_n_words"
            ],
        ),
        "runtime_seconds": round(runtime_seconds, 2),
    }

    output_root = Path(protocol["output_dir"])

    if arguments.limit is not None:
        output_root = output_root / f"smoke_{arguments.limit}"

    run_output_dir = output_root / selected_run["run_id"]
    run_output_dir.mkdir(parents=True, exist_ok=True)

    with (run_output_dir / "metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    with (run_output_dir / "config_used.yaml").open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            run_config,
            file,
            allow_unicode=True,
            sort_keys=False,
        )

    topic_model.get_topic_info().to_csv(
        run_output_dir / "topic_info.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Lưu topic và độ tin cậy của từng bài train.
    train_document_topics = build_document_topics(
        train_dataset,
        train_topics,
        train_probabilities,
        split="train",
    )

    # Lưu kết quả áp dụng model lên validation.
    validation_document_topics = build_document_topics(
        validation_dataset,
        validation_topics,
        validation_probabilities,
        split="val",
    )

    document_topics = pd.concat(
        [
            train_document_topics,
            validation_document_topics,
        ],
        ignore_index=True,
    )

    # Xuất từ khóa c-TF-IDF của từng topic.
    topic_keywords = build_topic_keywords(
        topic_model
    )

    # Chỉ dùng train để tính kích thước và mô tả topic.
    topic_summary = build_topic_summary(
        train_document_topics,
        topic_keywords,
    )

    document_topics.to_csv(
        run_output_dir / "document_topics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    topic_keywords.to_csv(
        run_output_dir / "topic_keywords.csv",
        index=False,
        encoding="utf-8-sig",
    )

    topic_summary.to_csv(
        run_output_dir / "topic_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    output_root.mkdir(parents=True, exist_ok=True)
    update_summary(
        output_root / "summary.csv",
        result,
    )

    artifact_dir = (
        Path("artifacts")
        / "bertopic"
        / "experiments"
        / selected_run["run_id"]
    )
    topic_model.save(
        artifact_dir / "model",
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=embedding_config["model_name"],
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Kết quả: {run_output_dir}")


if __name__ == "__main__":
    main()