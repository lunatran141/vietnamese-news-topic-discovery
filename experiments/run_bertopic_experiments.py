"""Chạy các thí nghiệm embedding và tham số BERTopic cho M14."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any
from umap import UMAP

import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer

from src.data.dataset_loader import load_split_dataset
from src.evaluation.bertopic_artifacts import (
    build_document_topics,
    build_embedding_projection,
    build_topic_keywords,
    build_topic_summary,
)
from src.evaluation.bertopic_metrics import (
    calculate_external_metrics,
    calculate_outlier_rate,
    calculate_topic_diversity,
)
from src.evaluation.lda_metrics import (
    calculate_coherence,
    prepare_coherence_corpus,
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
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help=(
            "Sau khi đã khóa cấu hình, đánh giá thêm trên test. "
            "Không dùng kết quả test để chọn run."
        ),
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
    config["dataset"]["text_variant"] = run_config.get(
        "input_text_column",
        config["dataset"]["text_variant"],
    )
    config["experiment"]["input_text_label"] = run_config.get(
        "input_text_label",
        "title_content",
    )

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
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
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

    dataset_config = run_config["dataset"]

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
    test_dataset = None
    if arguments.evaluate_test:
        test_dataset = load_split_dataset(
            dataset_config["path"],
            dataset_config["split_manifest_path"],
            dataset_config["test_split"],
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
        if test_dataset is not None:
            test_dataset = test_dataset.head(arguments.limit).copy()

    text_column = dataset_config["text_variant"]
    train_documents = prepare_documents(
        train_dataset,
        text_column,
    )
    validation_documents = prepare_documents(
        validation_dataset,
        text_column,
    )
    test_documents = (
        prepare_documents(test_dataset, text_column)
        if test_dataset is not None
        else []
    )

    embedding_config = run_config["embedding"]
    model_arguments: dict[str, Any] = {}

    if embedding_config.get("device"):
        model_arguments["device"] = embedding_config["device"]

    print(f"Run: {arguments.run_id}")
    print(f"Train: {len(train_documents)} bài")
    print(f"Validation: {len(validation_documents)} bài")
    print(f"Embedding: {embedding_config['model_name']}")
    print(
        "Đầu vào: "
        f"{selected_run.get('input_text_label', 'title_content')} "
        f"({text_column})"
    )

    pipeline_start = time.perf_counter()
    model_load_start = time.perf_counter()
    embedding_model = SentenceTransformer(
        embedding_config["model_name"],
        **model_arguments,
    )
    model_load_seconds = time.perf_counter() - model_load_start

    representation_start = time.perf_counter()
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
    test_embeddings = (
        encode_documents(
            embedding_model,
            test_documents,
            run_config,
        )
        if test_documents
        else None
    )
    representation_seconds = (
        time.perf_counter() - representation_start
    )

    model_init_start = time.perf_counter()
    topic_model = build_bertopic_model(
        run_config,
        embedding_model=embedding_model,
    )
    model_init_seconds = time.perf_counter() - model_init_start

    fit_start = time.perf_counter()
    train_topics, train_probabilities = fit_bertopic_model(
        topic_model,
        train_documents,
        train_embeddings,
    )
    fit_seconds = time.perf_counter() - fit_start

    validation_predict_start = time.perf_counter()
    validation_topics, validation_probabilities = (
        topic_model.transform(
            validation_documents,
            embeddings=validation_embeddings,
        )
    )
    validation_predict_seconds = (
        time.perf_counter() - validation_predict_start
    )

    test_topics = None
    test_probabilities = None
    test_predict_seconds = None
    if test_documents:
        test_predict_start = time.perf_counter()
        test_topics, test_probabilities = topic_model.transform(
            test_documents,
            embeddings=test_embeddings,
        )
        test_predict_seconds = time.perf_counter() - test_predict_start

    runtime_seconds = time.perf_counter() - pipeline_start

    evaluation_start = time.perf_counter()
    topic_words = topic_model.get_topics()
    topic_count = len(
        {
            int(topic)
            for topic in train_topics
            if int(topic) != -1
        }
    )

    validation_external_metrics: dict[str, float] = {}
    if "category_gold" in validation_dataset.columns:
        validation_external_metrics = calculate_external_metrics(
            validation_dataset["category_gold"].fillna("unknown"),
            validation_topics,
        )

    topic_word_lists = [
        [word for word, _ in words[: base_config["topic_representation"]["top_n_words"]]]
        for topic_id, words in topic_words.items()
        if int(topic_id) != -1
    ]
    coherence_texts, coherence_dictionary = prepare_coherence_corpus(
        validation_documents,
        topic_model.vectorizer_model,
    )
    validation_npmi_coherence = calculate_coherence(
        topic_word_lists,
        coherence_texts,
        coherence_dictionary,
        metric="c_npmi",
        processes=1,
    )

    test_external_metrics: dict[str, float] = {}
    test_npmi_coherence = None
    if test_dataset is not None and test_topics is not None:
        if "category_gold" in test_dataset.columns:
            test_external_metrics = calculate_external_metrics(
                test_dataset["category_gold"].fillna("unknown"),
                test_topics,
            )
        test_coherence_texts, test_coherence_dictionary = (
            prepare_coherence_corpus(
                test_documents,
                topic_model.vectorizer_model,
            )
        )
        test_npmi_coherence = calculate_coherence(
            topic_word_lists,
            test_coherence_texts,
            test_coherence_dictionary,
            metric="c_npmi",
            processes=1,
        )

    result: dict[str, Any] = {
        "run_id": selected_run["run_id"],
        "description": selected_run["description"],
        "embedding_model": embedding_config["model_name"],
        "input_text_column": text_column,
        "input_text_label": selected_run.get(
            "input_text_label",
            "title_content",
        ),
        "train_count": len(train_documents),
        "validation_count": len(validation_documents),
        "test_count": len(test_documents) if test_documents else None,
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
        "validation_adjusted_rand_index": (
            validation_external_metrics.get("adjusted_rand_index")
        ),
        "validation_normalized_mutual_info": (
            validation_external_metrics.get("normalized_mutual_info")
        ),
        "validation_npmi_coherence": validation_npmi_coherence,
        "test_outlier_rate": (
            calculate_outlier_rate(test_topics)
            if test_topics is not None
            else None
        ),
        "test_coverage": (
            1.0 - calculate_outlier_rate(test_topics)
            if test_topics is not None
            else None
        ),
        "test_adjusted_rand_index": test_external_metrics.get(
            "adjusted_rand_index"
        ),
        "test_normalized_mutual_info": test_external_metrics.get(
            "normalized_mutual_info"
        ),
        "test_npmi_coherence": test_npmi_coherence,
        "model_load_seconds": round(model_load_seconds, 2),
        "representation_seconds": round(
            representation_seconds, 2
        ),
        "model_init_seconds": round(model_init_seconds, 2),
        "fit_seconds": round(fit_seconds, 2),
        "validation_predict_seconds": round(
            validation_predict_seconds, 2
        ),
        "test_predict_seconds": (
            round(test_predict_seconds, 2)
            if test_predict_seconds is not None
            else None
        ),
        "runtime_seconds": round(runtime_seconds, 2),
        "runtime_scope": (
            "model_load + representation + model_init + fit + "
            "validation_prediction"
            + (" + test_prediction" if test_documents else "")
        ),
    }
    result["evaluation_seconds"] = round(
        time.perf_counter() - evaluation_start,
        2,
    )

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
    test_document_topics = None
    if test_dataset is not None and test_topics is not None:
        test_document_topics = build_document_topics(
            test_dataset,
            test_topics,
            test_probabilities,
            split="test",
        )

    document_frames = [
        train_document_topics,
        validation_document_topics,
    ]
    if test_document_topics is not None:
        document_frames.append(test_document_topics)
    document_topics = pd.concat(
        document_frames,
        ignore_index=True,
    )

        # UMAP này chỉ phục vụ trực quan hóa embedding.
    # Nó độc lập với UMAP bên trong pipeline BERTopic,
    # nên không làm thay đổi kết quả phân cụm.
    visualization_umap = UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )

    train_embeddings_2d = (
        visualization_umap.fit_transform(
            train_embeddings
        )
    )

    validation_embeddings_2d = (
        visualization_umap.transform(
            validation_embeddings
        )
    )

    train_projection = build_embedding_projection(
        article_ids=train_dataset["article_id"],
        split="train",
        topics=train_topics,
        embeddings_2d=train_embeddings_2d,
    )

    validation_projection = build_embedding_projection(
        article_ids=validation_dataset["article_id"],
        split="val",
        topics=validation_topics,
        embeddings_2d=validation_embeddings_2d,
    )
    test_projection = None
    if test_dataset is not None and test_topics is not None:
        test_embeddings_2d = visualization_umap.transform(test_embeddings)
        test_projection = build_embedding_projection(
            article_ids=test_dataset["article_id"],
            split="test",
            topics=test_topics,
            embeddings_2d=test_embeddings_2d,
        )

    projection_frames = [train_projection, validation_projection]
    if test_projection is not None:
        projection_frames.append(test_projection)
    embedding_projection = pd.concat(
        projection_frames,
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

    embedding_projection.to_csv(
        run_output_dir / "embedding_projection.csv",
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
