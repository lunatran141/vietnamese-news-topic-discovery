
"""Run M07 Bag-of-Words + LDA on Dataset v1."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.sparse import save_npz

from src.data.dataset_loader import load_split_dataset
from src.features.article_text import build_article_text
from src.features.bow import (
    fit_bow_pipeline,
    transform_bow_pipeline,
)
from src.models.lda_pipeline import (
    fit_lda_pipeline,
    transform_lda_pipeline,
    save_lda_model,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run M07 BoW + LDA pipeline"
    )
    parser.add_argument(
        "--config",
        default="configs/lda.yaml",
    )
    return parser.parse_args()


def save_document_topics(
    df: pd.DataFrame,
    distributions: np.ndarray,
    path: Path,
):
    """Save topic probabilities with article IDs."""
    result = pd.DataFrame(
        distributions,
        columns=[
            f"topic_{i}"
            for i in range(distributions.shape[1])
        ],
    )

    result.insert(
        0,
        "article_id",
        df["article_id"].astype(str).to_numpy(),
    )

    result["dominant_topic"] = distributions.argmax(axis=1)
    result["dominant_probability"] = distributions.max(axis=1)

    result.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    args = parse_args()
    start = time.perf_counter()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_cfg = config["dataset"]
    artifacts_dir = Path(config["artifacts"]["output_dir"])
    results_dir = Path(config["results"]["output_dir"])

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Dataset v1...")

    train_df = load_split_dataset(
        dataset_cfg["path"],
        dataset_cfg["split_manifest_path"],
        dataset_cfg["train_split"],
    )

    val_df = load_split_dataset(
        dataset_cfg["path"],
        dataset_cfg["split_manifest_path"],
        dataset_cfg["val_split"],
    )

    # Test set remains locked for final evaluation.
    train_ids = set(train_df["article_id"].astype(str))
    val_ids = set(val_df["article_id"].astype(str))

    if train_ids & val_ids:
        raise ValueError("Train/Validation ID leakage detected")

    text_variant = dataset_cfg["text_variant"]

    train_texts = build_article_text(
        train_df, text_variant
    ).tolist()

    val_texts = build_article_text(
        val_df, text_variant
    ).tolist()

    print(f"Train documents: {len(train_texts)}")
    print(f"Validation documents: {len(val_texts)}")

    print("Fitting Bag-of-Words on Train...")

    vectorizer, train_bow = fit_bow_pipeline(
        train_texts,
        config["vectorizer"],
    )

    val_bow = transform_bow_pipeline(
        vectorizer,
        val_texts,
    )

    print(f"Vocabulary size: {train_bow.shape[1]}")

    print("Training LDA on Train...")

    model = fit_lda_pipeline(
        train_bow,
        config["lda"],
    )

    train_topics = transform_lda_pipeline(
        model, train_bow
    )

    val_topics = transform_lda_pipeline(
        model, val_bow
    )

    print("Saving artifacts...")

    joblib.dump(
        vectorizer,
        artifacts_dir / "bow_vectorizer.joblib",
    )

    save_lda_model(
        model,
        artifacts_dir / "lda_model.joblib",
    )

    save_npz(
        artifacts_dir / "bow_train.npz",
        train_bow,
    )

    save_npz(
        artifacts_dir / "bow_val.npz",
        val_bow,
    )

    feature_names = vectorizer.get_feature_names_out()

    with open(
        artifacts_dir / "feature_names.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            feature_names.tolist(),
            f,
            ensure_ascii=False,
            indent=2,
        )

    save_document_topics(
        train_df,
        train_topics,
        results_dir / "document_topics_train.csv",
    )

    save_document_topics(
        val_df,
        val_topics,
        results_dir / "document_topics_val.csv",
    )

    # Topic-word distribution, normalized per topic.
    topic_word = model.components_.copy()
    topic_word /= topic_word.sum(axis=1, keepdims=True)

    np.save(
        artifacts_dir / "topic_word_distribution.npy",
        topic_word,
    )

    duration = round(time.perf_counter() - start, 2)

    metadata = {
        "task": "M07",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset_version": dataset_cfg["data_version"],
        "dataset_path": dataset_cfg["path"],
        "preprocessing_variant": dataset_cfg[
            "preprocessing_variant"
        ],
        "text_variant": text_variant,
        "split_manifest_path": dataset_cfg[
            "split_manifest_path"
        ],
        "train_documents": len(train_df),
        "val_documents": len(val_df),
        "vocab_size": len(feature_names),
        "n_topics": int(model.n_components),
        "random_state": config["lda"]["random_state"],
        "train_bow_shape": list(train_bow.shape),
        "val_bow_shape": list(val_bow.shape),
        "train_zero_bow_docs": int(
            np.sum(train_bow.getnnz(axis=1) == 0)
        ),
        "val_zero_bow_docs": int(
            np.sum(val_bow.getnnz(axis=1) == 0)
        ),
        "runtime_seconds": duration,
        "test_set_used": False,
        "config": config,
    }

    with open(
        artifacts_dir / "run_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nM07 completed successfully!")
    print(f"Train topic shape: {train_topics.shape}")
    print(f"Val topic shape: {val_topics.shape}")
    print(f"Runtime: {duration} seconds")
    print(f"Artifacts: {artifacts_dir}")
    print(f"Results: {results_dir}")


if __name__ == "__main__":
    main()
