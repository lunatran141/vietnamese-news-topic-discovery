"""Chạy BERTopic baseline trên tập train chính thức của Dataset v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

from src.data.dataset_loader import load_split_dataset
from src.models.bertopic_pipeline import (
    build_bertopic_model,
    encode_documents,
    fit_bertopic_model,
    load_bertopic_config,
)


def parse_arguments() -> argparse.Namespace:
    """Đọc tham số dòng lệnh cho thí nghiệm."""
    parser = argparse.ArgumentParser(
        description="Chạy BERTopic baseline cho task M13."
    )
    parser.add_argument(
        "--config",
        default="configs/bertopic.yaml",
        help="Đường dẫn tới file cấu hình BERTopic.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số bài để chạy thử.",
    )
    return parser.parse_args()


def main() -> None:
    """Chạy pipeline từ Dataset v1 đến kết quả topic."""
    arguments = parse_arguments()
    config = load_bertopic_config(arguments.config)

    dataset_config = config["dataset"]
    embedding_config = config["embedding"]
    experiment_config = config["experiment"]

    dataset = load_split_dataset(
        dataset_config["path"],
        dataset_config["split_manifest_path"],
        dataset_config["fit_split"],
    )

    if arguments.limit is not None:
        if arguments.limit <= 0:
            raise ValueError("--limit phải lớn hơn 0.")
        dataset = dataset.head(arguments.limit).copy()

    text_column = dataset_config["text_variant"]

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

    print(f"Số bài dùng để fit: {len(documents)}")
    print(f"Embedding model: {embedding_config['model_name']}")

    model_arguments = {}

    if embedding_config.get("device"):
        model_arguments["device"] = embedding_config["device"]

    embedding_model = SentenceTransformer(
        embedding_config["model_name"],
        **model_arguments,
    )

    embeddings = encode_documents(
        embedding_model,
        documents,
        config,
    )

    topic_model = build_bertopic_model(
        config,
        embedding_model=embedding_model,
    )

    topics, probabilities = fit_bertopic_model(
        topic_model,
        documents,
        embeddings,
    )

    output_dir = Path(experiment_config["output_dir"])

    # Kết quả chạy thử được tách khỏi kết quả baseline chính thức.
    if arguments.limit is not None:
        output_dir = output_dir / f"smoke_{arguments.limit}"

    output_dir.mkdir(parents=True, exist_ok=True)

    document_topics = pd.DataFrame(
        {
            "article_id": dataset["article_id"].tolist(),
            "topic": topics,
        }
    )

    if probabilities is not None and probabilities.ndim == 1:
        document_topics["topic_probability"] = probabilities

    document_topics.to_parquet(
        output_dir / "document_topics.parquet",
        index=False,
    )

    topic_model.get_topic_info().to_csv(
        output_dir / "topic_info.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "run_id": experiment_config["run_id"],
        "dataset_version": dataset_config["version"],
        "fit_split": dataset_config["fit_split"],
        "document_count": len(documents),
        "embedding_model": embedding_config["model_name"],
        "config_path": str(arguments.config),
    }

    with (output_dir / "run_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    if experiment_config["save_model"]:
        topic_model.save(
            output_dir / "model",
            serialization="safetensors",
            save_ctfidf=True,
            save_embedding_model=embedding_config["model_name"],
        )

    print(f"Đã lưu kết quả tại: {output_dir}")


if __name__ == "__main__":
    main()