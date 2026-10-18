
"""Task 3.7: Final LDA evaluation for comparison with NMF."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from experiments.nmf.metrics import npmi_coherence
from src.data.dataset_loader import load_split_dataset
from src.features.article_text import build_article_text
from src.features.bow import fit_bow_pipeline, transform_bow_pipeline
from src.models.lda_pipeline import fit_lda_pipeline, transform_lda_pipeline
from src.evaluation.lda_metrics import extract_topic_words, topic_diversity


def load_config(config_path):
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    with open(config["base_config"], encoding="utf-8") as file:
        base_config = yaml.safe_load(file)

    return config, base_config


def validate_splits(train, val, test=None):
    frames = {"train": train, "val": val}
    if test is not None:
        frames["test"] = test

    seen = set()
    for name, frame in frames.items():
        ids = frame["article_id"]
        if frame.empty or ids.isna().any() or ids.duplicated().any():
            raise ValueError(f"Invalid article_id in {name}")

        current = set(ids.astype(str))
        if current & seen:
            raise ValueError(f"Data leakage detected in {name}")
        seen.update(current)


def prepare(config, base):
    data = config["dataset"]
    original = base["dataset"]

    for key in ("path", "split_manifest_path", "text_variant"):
        if data[key] != original[key]:
            raise ValueError(f"Dataset config mismatch: {key}")

    if data["data_version"] != "v1.0":
        raise ValueError("Expected Dataset v1.0")

    if data["preprocessing_variant"] != "p2_segmented":
        raise ValueError("Expected P2 Segmented for LDA")

    k = int(config["selection"]["selected_n_topics"])
    if k != 15:
        raise ValueError("K differs from proposed M08/M09 configuration")

    train = load_split_dataset(
        data["path"], data["split_manifest_path"], "train"
    )
    val = load_split_dataset(
        data["path"], data["split_manifest_path"], "val"
    )

    validate_splits(train, val)

    print("=== TASK 3.7: LDA PREPARATION ===")
    print("Dataset: v1.0")
    print("Preprocessing: P2 Segmented")
    print(f"Train: {len(train)} documents")
    print(f"Validation: {len(val)} documents")
    print(f"Proposed K: {k}")
    print(f"Seed: {config['model']['random_state']}")
    print(f"Approval: {config['selection']['approval_status']}")
    print("Test set: NOT LOADED")
    print("Preparation completed.")

    return train, val


def evaluate_test(model, vectorizer, test_bow, gold, top_n):
    distributions = transform_lda_pipeline(model, test_bow)
    assignments = distributions.argmax(axis=1)

    if len(gold) != len(assignments) or pd.isna(gold).any():
        raise ValueError("Invalid test gold labels")

    topics = extract_topic_words(
        model, vectorizer.get_feature_names_out(), top_n
    )

    word_ids = [
        [vectorizer.vocabulary_[word] for word in words]
        for words in topics
    ]

    # Common NPMI definition reused from the NMF implementation.
    common_npmi, per_topic_npmi = npmi_coherence(
        test_bow, word_ids
    )

    metrics = {
        "n_topics": int(model.n_components),
        "npmi_coherence": float(common_npmi),
        "npmi_definition": "document_binary_common",
        "topic_diversity": float(topic_diversity(topics)),
        "adjusted_rand_index": float(
            adjusted_rand_score(gold, assignments)
        ),
        "normalized_mutual_info": float(
            normalized_mutual_info_score(gold, assignments)
        ),
        "test_documents": int(len(gold)),
    }
    return metrics, distributions, topics, per_topic_npmi


def finalize(config, base, train, val):
    if config["selection"]["approval_status"] != "approved":
        raise ValueError("K=15 has not been approved by the team")

    if config["safety"]["allow_test"] is not True:
        raise ValueError("Test evaluation is locked in YAML")

    artifact_dir = Path(config["outputs"]["artifacts_dir"])
    result_dir = Path(config["outputs"]["results_dir"])

    if artifact_dir.exists() or result_dir.exists():
        raise FileExistsError(
            "Final output already exists. Refusing to overwrite it."
        )

    data = config["dataset"]
    test = load_split_dataset(
        data["path"], data["split_manifest_path"], "test"
    )
    validate_splits(train, val, test)

    development = pd.concat([train, val], ignore_index=True)

    development_texts = build_article_text(
        development, data["text_variant"]
    ).tolist()
    test_texts = build_article_text(
        test, data["text_variant"]
    ).tolist()

    vectorizer, development_bow = fit_bow_pipeline(
        development_texts, base["vectorizer"]
    )
    test_bow = transform_bow_pipeline(vectorizer, test_texts)

    lda_config = copy.deepcopy(base["lda"])
    lda_config["n_components"] = int(
        config["selection"]["selected_n_topics"]
    )
    lda_config["random_state"] = int(
        config["model"]["random_state"]
    )

    model = fit_lda_pipeline(development_bow, lda_config)

    metrics, distributions, topics, topic_npmi = evaluate_test(
        model,
        vectorizer,
        test_bow,
        test["category_gold"],
        int(config["model"]["top_n_words"]),
    )

    artifact_dir.mkdir(parents=True)
    result_dir.mkdir(parents=True)

    joblib.dump(
        {"vectorizer": vectorizer, "lda_model": model},
        artifact_dir / "lda_pipeline.joblib",
    )

    metadata = {
        "task": "3.7",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": "v1.0",
        "preprocessing_variant": "p2_segmented",
        "fit_split": "train+val",
        "evaluation_split": "test",
        "train_documents": len(train),
        "val_documents": len(val),
        "test_documents": len(test),
        "selected_n_topics": lda_config["n_components"],
        "random_state": lda_config["random_state"],
        "selection_source": config["selection"]["selection_source"],
        "npmi_definition": "document_binary_common",
    }

    (artifact_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "config.json").write_text(
        json.dumps(
            {"final": config, "base": base, "effective_lda": lda_config},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (result_dir / "test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame([
        {
            "topic_id": index,
            "keywords": ", ".join(words),
            "npmi_common": topic_npmi[index],
        }
        for index, words in enumerate(topics)
    ]).to_csv(
        result_dir / "topic_keywords.csv",
        index=False,
        encoding="utf-8-sig",
    )

    document_topics = pd.DataFrame(
        distributions,
        columns=[
            f"topic_{index}"
            for index in range(distributions.shape[1])
        ],
    )
    document_topics.insert(
        0, "article_id", test["article_id"].astype(str).to_numpy()
    )
    document_topics["dominant_topic"] = distributions.argmax(axis=1)
    document_topics["dominant_probability"] = distributions.max(axis=1)

    document_topics.to_csv(
        result_dir / "document_topics_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("LDA Final Test completed.")
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/lda_final.yaml"
    )
    parser.add_argument(
        "--stage", choices=["prepare", "finalize"], default="prepare"
    )
    parser.add_argument("--confirm-test", action="store_true")
    args = parser.parse_args()

    config, base = load_config(args.config)

    if args.stage == "finalize":
        if not args.confirm_test:
            raise ValueError("Final Test requires --confirm-test")
        if config["selection"]["approval_status"] != "approved":
            raise ValueError("K has not been approved")
        if config["safety"]["allow_test"] is not True:
            raise ValueError("Test remains locked")

    train, val = prepare(config, base)

    if args.stage == "finalize":
        finalize(config, base, train, val)


if __name__ == "__main__":
    main()
