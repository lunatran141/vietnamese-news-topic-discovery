
"""Supplementary LDA Development evaluation.

Fit: Train
Evaluate: Validation
Model: K=15, seed=42

Does not load Test or overwrite approved Final outputs.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from experiments.lda.run_lda_final import (
    load_config,
    prepare,
    validate_splits,
)
from experiments.nmf.metrics import npmi_coherence
from src.features.article_text import build_article_text
from src.features.bow import (
    fit_bow_pipeline,
    transform_bow_pipeline,
)
from src.models.lda_pipeline import (
    fit_lda_pipeline,
    transform_lda_pipeline,
)
from src.evaluation.lda_metrics import (
    extract_topic_words,
    topic_diversity,
)


CONFIG_PATH = Path("configs/lda_final.yaml")
OUTPUT_DIR = Path("results/lda/validation_supplement")
OUTPUT_FILE = OUTPUT_DIR / "validation_metrics.json"


def main():
    # Do not overwrite previous evaluation results.
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Evaluation already exists: {OUTPUT_FILE}"
        )

    config, base = load_config(CONFIG_PATH)

    if config["selection"]["approval_status"] != "approved":
        raise ValueError("LDA K selection is not approved")

    k = int(config["selection"]["selected_n_topics"])
    seed = int(config["model"]["random_state"])
    top_n = int(config["model"]["top_n_words"])

    if k != 15 or seed != 42:
        raise ValueError("Expected approved LDA K=15, seed=42")

    # prepare() loads Train and Validation only.
    train, val = prepare(config, base)
    validate_splits(train, val)

    data_cfg = config["dataset"]

    train_texts = build_article_text(
        train, data_cfg["text_variant"]
    ).tolist()

    val_texts = build_article_text(
        val, data_cfg["text_variant"]
    ).tolist()

    # Fit the vocabulary on Train only.
    vectorizer, train_bow = fit_bow_pipeline(
        train_texts,
        base["vectorizer"],
    )

    val_bow = transform_bow_pipeline(
        vectorizer,
        val_texts,
    )

    lda_cfg = copy.deepcopy(base["lda"])
    lda_cfg["n_components"] = k
    lda_cfg["random_state"] = seed

    # Fresh Development fit, not the approved Final model.
    model = fit_lda_pipeline(train_bow, lda_cfg)

    val_distributions = transform_lda_pipeline(
        model, val_bow
    )

    if val_distributions.shape != (len(val), k):
        raise ValueError("Invalid Validation distribution shape")

    if not np.isfinite(val_distributions).all():
        raise ValueError("Invalid topic distributions")

    assignments = val_distributions.argmax(axis=1)

    gold = val["category_gold"]

    if gold.isna().any() or len(gold) != len(assignments):
        raise ValueError("Invalid Validation gold labels")

    topics = extract_topic_words(
        model,
        vectorizer.get_feature_names_out(),
        top_n,
    )

    word_ids = [
        [vectorizer.vocabulary_[word] for word in words]
        for words in topics
    ]

    common_npmi, per_topic_npmi = npmi_coherence(
        val_bow,
        word_ids,
    )

    perplexity = float(model.perplexity(val_bow))

    metrics = {
        "run_id": "lda_k15_seed42_validation_supplement",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset_version": "v1.0",
        "preprocessing_variant": "p2_segmented",
        "n_topics": k,
        "random_state": seed,
        "fit_split": "train",
        "evaluation_split": "val",
        "train_documents": int(len(train)),
        "validation_documents": int(len(val)),
        "npmi_definition": "document_binary_common",
        "validation_ari": float(
            adjusted_rand_score(gold, assignments)
        ),
        "validation_nmi": float(
            normalized_mutual_info_score(gold, assignments)
        ),
        "validation_npmi": float(common_npmi),
        "validation_perplexity": perplexity,
        "topic_diversity": float(
            topic_diversity(topics)
        ),
        "per_topic_npmi": [
            float(score) for score in per_topic_npmi
        ],
        "notes": (
            "Development-only supplementary evaluation. "
            "LDA fitted on Train and evaluated on Validation. "
            "Fixed approved K=15 and seed=42. "
            "No Test data used. "
            "Does not replace M08 or Final Test metrics."
        ),
    }

    for name in (
        "validation_ari",
        "validation_nmi",
        "validation_npmi",
        "validation_perplexity",
        "topic_diversity",
    ):
        if not np.isfinite(metrics[name]):
            raise ValueError(f"Invalid metric: {name}")

    if metrics["validation_perplexity"] <= 0:
        raise ValueError("Perplexity must be positive")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Exclusive creation: refuse to overwrite an existing file.
    with OUTPUT_FILE.open("x", encoding="utf-8") as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== LDA VALIDATION SUPPLEMENT ===")
    print(f"Train documents: {len(train)}")
    print(f"Validation documents: {len(val)}")
    print(f"K: {k}, Seed: {seed}")

    for name in (
        "validation_ari",
        "validation_nmi",
        "validation_npmi",
        "validation_perplexity",
        "topic_diversity",
    ):
        print(f"{name}: {metrics[name]:.6f}")

    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
