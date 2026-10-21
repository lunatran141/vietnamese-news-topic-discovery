"""Đánh giá một lần cấu hình BERTopic E1 đã khóa trên tập test."""

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
from src.evaluation.bertopic_artifacts import build_document_topics
from src.evaluation.bertopic_metrics import (
    calculate_external_metrics,
    calculate_outlier_rate,
)
from src.models.bertopic_pipeline import (
    build_bertopic_model,
    encode_documents,
    fit_bertopic_model,
    load_bertopic_config,
)


RUN_ID = "e1_vietnamese_sbert"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Đánh giá cấu hình E1 đã khóa trên test."
    )
    parser.add_argument(
        "--confirm-test-evaluation",
        action="store_true",
        help="Xác nhận chỉ đánh giá, không dùng test để tuning.",
    )
    parser.add_argument(
        "--base-config",
        default="configs/bertopic.yaml",
    )
    parser.add_argument(
        "--experiment-config",
        default="configs/bertopic_experiments.yaml",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError("Cấu hình phải là dictionary.")
    return value


def main() -> None:
    arguments = parse_arguments()
    if not arguments.confirm_test_evaluation:
        raise ValueError(
            "Test đang khóa. Chỉ thêm --confirm-test-evaluation sau khi "
            "E1 và tiêu chí đánh giá đã được chốt."
        )

    base_config = load_bertopic_config(arguments.base_config)
    experiment_config = load_yaml(arguments.experiment_config)
    selected = next(
        run for run in experiment_config["runs"] if run["run_id"] == RUN_ID
    )
    run_config = copy.deepcopy(base_config)
    for section in ("embedding", "umap", "hdbscan"):
        run_config[section] = copy.deepcopy(selected[section])

    dataset_config = base_config["dataset"]
    train_dataset = load_split_dataset(
        dataset_config["path"],
        dataset_config["split_manifest_path"],
        dataset_config["fit_split"],
    )
    validation_dataset = load_split_dataset(
        dataset_config["path"],
        dataset_config["split_manifest_path"],
        dataset_config["validation_split"],
    )
    final_train_dataset = pd.concat(
        [train_dataset, validation_dataset],
        ignore_index=True,
    )
    test_dataset = load_split_dataset(
        dataset_config["path"],
        dataset_config["split_manifest_path"],
        dataset_config["test_split"],
    )
    train_documents = (
        final_train_dataset[dataset_config["text_variant"]]
        .fillna("")
        .astype(str)
        .str.strip()
        .tolist()
    )
    test_documents = (
        test_dataset[dataset_config["text_variant"]]
        .fillna("")
        .astype(str)
        .str.strip()
        .tolist()
    )

    embedding_config = selected["embedding"]
    model_arguments: dict[str, Any] = {}
    if embedding_config.get("device"):
        model_arguments["device"] = embedding_config["device"]

    load_start = time.perf_counter()
    embedding_model = SentenceTransformer(
        embedding_config["model_name"],
        **model_arguments,
    )
    model_load_seconds = time.perf_counter() - load_start

    representation_start = time.perf_counter()
    train_embeddings = encode_documents(
        embedding_model, train_documents, run_config
    )
    test_embeddings = encode_documents(
        embedding_model, test_documents, run_config
    )
    representation_seconds = time.perf_counter() - representation_start

    model = build_bertopic_model(
        run_config,
        embedding_model=embedding_model,
    )
    fit_start = time.perf_counter()
    fit_bertopic_model(model, train_documents, train_embeddings)
    fit_seconds = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    topics, probabilities = model.transform(
        test_documents,
        embeddings=test_embeddings,
    )
    test_predict_seconds = time.perf_counter() - predict_start

    metrics = calculate_external_metrics(
        test_dataset["category_gold"].fillna("unknown"),
        topics,
    )
    metrics.update(
        {
            "final_train_count": len(train_documents),
            "test_count": len(test_documents),
            "test_outlier_rate": calculate_outlier_rate(topics),
            "test_coverage": 1.0 - calculate_outlier_rate(topics),
            "test_adjusted_rand_index": metrics.pop("adjusted_rand_index"),
            "test_normalized_mutual_info": metrics.pop(
                "normalized_mutual_info"
            ),
            "test_model_load_seconds": round(model_load_seconds, 2),
            "test_representation_seconds": round(
                representation_seconds, 2
            ),
            "final_fit_seconds": round(fit_seconds, 2),
            "test_predict_seconds": round(test_predict_seconds, 2),
            "final_runtime_seconds": round(
                model_load_seconds
                + representation_seconds
                + fit_seconds
                + test_predict_seconds,
                2,
            ),
            "final_runtime_scope": (
                "model_load + train_validation_and_test_representation + "
                "train_validation_fit + test_prediction"
            ),
        }
    )

    output_dir = Path("results/bertopic_experiments") / RUN_ID
    test_assignments = build_document_topics(
        test_dataset,
        topics,
        probabilities,
        split="test",
    )
    test_assignments.to_csv(
        output_dir / "test_document_topics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("r", encoding="utf-8") as file:
        combined_metrics = json.load(file)
    combined_metrics.update(metrics)
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(combined_metrics, file, ensure_ascii=False, indent=2)

    document_topics_path = output_dir / "document_topics.csv"
    document_topics = pd.read_csv(document_topics_path, encoding="utf-8-sig")
    document_topics = document_topics.loc[document_topics["split"] != "test"]
    pd.concat([document_topics, test_assignments], ignore_index=True).to_csv(
        document_topics_path,
        index=False,
        encoding="utf-8-sig",
    )
    final_model_path = Path("artifacts/bertopic/final") / RUN_ID / "model"
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(
        final_model_path,
        serialization="pickle",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
