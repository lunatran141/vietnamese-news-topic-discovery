
"""Supplementary M09 per-seed metrics on Validation.

Fits K=15 LDA for the five approved M09 seeds.
Uses the M09 Train-only BoW protocol.
Never loads Test or modifies original M09 results.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from experiments.lda.run_m09_lda_stability import (
    load_yaml,
    validate_config,
    load_data_and_bow,
)
from experiments.nmf.metrics import npmi_coherence
from src.data.dataset_loader import load_split_dataset
from src.models.lda_pipeline import (
    fit_lda_pipeline,
    transform_lda_pipeline,
)
from src.evaluation.lda_metrics import (
    extract_topic_words,
    topic_diversity,
)

CONFIG_PATH = Path("configs/lda_stability.yaml")
OUTPUT_PATH = Path("results/lda/m09/stability_per_seed.csv")


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {OUTPUT_PATH}"
        )

    config = load_yaml(CONFIG_PATH)
    base = load_yaml(config["base_config"])

    k, top_n, seeds, reference_seed = validate_config(
        config, base
    )

    if k != 15 or seeds != [42, 52, 62, 72, 82]:
        raise ValueError("Unexpected M09 configuration")

    train_bow, val_bow, vocabulary = load_data_and_bow(
        config, base
    )

    val_df = load_split_dataset(
        config["dataset"]["path"],
        config["dataset"]["split_manifest_path"],
        config["dataset"]["val_split"],
    )

    if len(val_df) != val_bow.shape[0]:
        raise ValueError("Validation row count mismatch")

    if val_df["article_id"].isna().any():
        raise ValueError("Missing Validation article_id")

    if val_df["article_id"].duplicated().any():
        raise ValueError("Duplicate Validation article_id")

    gold = val_df["category_gold"]

    if gold.isna().any():
        raise ValueError("Missing Validation labels")

    runtime_path = (
        Path(config["outputs"]["results_dir"])
        / "runtime_by_seed.csv"
    )

    runtime_df = pd.read_csv(runtime_path)

    expected_columns = {
        "seed",
        "k",
        "runtime_seconds",
    }
    missing = expected_columns - set(runtime_df.columns)

    if missing:
        raise ValueError(
            f"Missing runtime columns: {sorted(missing)}"
        )

    runtime_df = runtime_df.loc[
        runtime_df["k"] == k
    ].copy()

    if (
        len(runtime_df) != len(seeds)
        or runtime_df["seed"].duplicated().any()
        or set(runtime_df["seed"]) != set(seeds)
    ):
        raise ValueError("M09 runtime seeds mismatch")

    runtime_map = dict(
        zip(
            runtime_df["seed"],
            runtime_df["runtime_seconds"],
        )
    )

    rows = []

    for seed in seeds:
        print(f"\nEvaluating K={k}, seed={seed}", flush=True)

        lda_cfg = dict(base["lda"])
        lda_cfg["n_components"] = k
        lda_cfg["random_state"] = seed

        model = fit_lda_pipeline(train_bow, lda_cfg)

        distributions = transform_lda_pipeline(
            model, val_bow
        )

        if distributions.shape != (len(val_df), k):
            raise ValueError("Invalid topic distribution shape")

        if not np.isfinite(distributions).all():
            raise ValueError("Non-finite topic distribution")

        if not np.allclose(
            distributions.sum(axis=1), 1.0, atol=1e-6
        ):
            raise ValueError(
                "Topic probabilities must sum to one"
            )

        assignments = distributions.argmax(axis=1)

        topics = extract_topic_words(
            model,
            vocabulary,
            top_n,
        )

        word_ids = [
            [
                int(model_word_id)
                for model_word_id in (
                    feature_index[word] for word in topic
                )
            ]
            for topic in topics
        ]

        npmi, _ = npmi_coherence(val_bow, word_ids)

        rows.append({
            "seed": int(seed),
            "topic_count": int(k),
            "ari": float(
                adjusted_rand_score(gold, assignments)
            ),
            "nmi": float(
                normalized_mutual_info_score(
                    gold, assignments
                )
            ),
            "npmi": float(npmi),
            "topic_diversity": float(
                topic_diversity(topics)
            ),
            "runtime_seconds": float(runtime_map[seed]),
        })

    result = pd.DataFrame(rows)

    if not np.isfinite(
        result[
            [
                "ari",
                "nmi",
                "npmi",
                "topic_diversity",
                "runtime_seconds",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite evaluation metrics")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Exclusive creation prevents accidental replacement.
    with OUTPUT_PATH.open(
        "x", encoding="utf-8-sig", newline=""
    ) as output:
        result.to_csv(output, index=False)

    print("\n=== M09 PER-SEED METRICS ===")
    print(result.to_string(index=False))
    print(f"\nSaved: {OUTPUT_PATH}")
    print(
        "Protocol: Train -> Validation; "
        "NPMI=document_binary_common"
    )
    print(
        "Runtime source: original M09 runtime_by_seed.csv"
    )


if __name__ == "__main__":
    # Preserve the exact vocabulary ordering from M09.
    config_for_vocab = load_yaml(CONFIG_PATH)
    base_for_vocab = load_yaml(
        config_for_vocab["base_config"]
    )

    # The BoW vocabulary is deterministic and fits Train only.
    from src.features.bow import fit_bow_pipeline
    from src.features.article_text import build_article_text

    train_for_vocab = load_split_dataset(
        config_for_vocab["dataset"]["path"],
        config_for_vocab["dataset"]["split_manifest_path"],
        config_for_vocab["dataset"]["train_split"],
    )

    vectorizer, _ = fit_bow_pipeline(
        build_article_text(
            train_for_vocab,
            config_for_vocab["dataset"]["text_variant"],
        ).tolist(),
        base_for_vocab["vectorizer"],
    )

    feature_index = vectorizer.vocabulary_
    main()
