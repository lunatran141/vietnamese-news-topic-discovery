
"""Extended end-to-end runtime benchmark for LDA K=15.

Run:
    python -m experiments.lda.benchmark_lda_end_to_end --repeats 5

This benchmark does not overwrite Final LDA artifacts.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from experiments.lda.run_lda_final import (
    load_config,
    validate_splits,
)
from experiments.nmf.metrics import npmi_coherence
from src.data.dataset_loader import load_split_dataset
from src.evaluation.lda_metrics import (
    extract_topic_words,
    topic_diversity,
)
from src.features.article_text import build_article_text
from src.features.bow import (
    fit_bow_pipeline,
    transform_bow_pipeline,
)
from src.models.lda_pipeline import (
    fit_lda_pipeline,
    transform_lda_pipeline,
)


CONFIG_PATH = Path("configs/lda_final.yaml")
OUTPUT_DIR = Path("results/lda/runtime_end_to_end")

SCENARIOS = [
    ("development", "train", "validation"),
    ("final", "train+val", "test"),
]


def timed(func, *args):
    start = time.perf_counter()
    result = func(*args)
    elapsed = time.perf_counter() - start
    return result, elapsed


def load_inputs(data_cfg, fit_split, predict_split):
    dataset_path = data_cfg["path"]
    manifest_path = data_cfg["split_manifest_path"]

    train = load_split_dataset(
        dataset_path, manifest_path, "train"
    )

    val = load_split_dataset(
        dataset_path, manifest_path, "val"
    )

    test = None
    if predict_split == "test":
        test = load_split_dataset(
            dataset_path, manifest_path, "test"
        )

    validate_splits(train, val, test)

    if fit_split == "train":
        development = train
    else:
        development = pd.concat(
            [train, val], ignore_index=True
        )

    evaluation = (
        val if predict_split == "validation" else test
    )

    train_texts = build_article_text(
        development,
        data_cfg["text_variant"],
    ).tolist()

    evaluation_texts = build_article_text(
        evaluation,
        data_cfg["text_variant"],
    ).tolist()

    gold = evaluation["category_gold"]

    if gold.isna().any():
        raise ValueError("Missing category_gold labels")

    return (
        train_texts,
        evaluation_texts,
        gold,
        len(development),
        len(evaluation),
    )


def evaluate_metrics(
    distributions,
    model,
    vectorizer,
    evaluation_bow,
    gold,
    top_n,
):
    topics = extract_topic_words(
        model,
        vectorizer.get_feature_names_out(),
        top_n,
    )

    word_ids = [
        [
            vectorizer.vocabulary_[word]
            for word in topic
        ]
        for topic in topics
    ]

    npmi, _ = npmi_coherence(
        evaluation_bow,
        word_ids,
    )

    assignments = distributions.argmax(axis=1)

    metrics = {
        "npmi_coherence": float(npmi),
        "npmi_definition": "document_binary_common",
        "topic_diversity": float(
            topic_diversity(topics)
        ),
        "adjusted_rand_index": float(
            adjusted_rand_score(gold, assignments)
        ),
        "normalized_mutual_info": float(
            normalized_mutual_info_score(
                gold, assignments
            )
        ),
    }

    return topics, assignments, metrics


def save_outputs(
    output_dir,
    vectorizer,
    model,
    distributions,
    assignments,
    topics,
    metrics,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "lda_model": model,
        },
        output_dir / "lda_pipeline.joblib",
    )

    pd.DataFrame(distributions).to_csv(
        output_dir / "document_topics.csv",
        index=False,
    )

    pd.DataFrame({
        "document_index": range(len(assignments)),
        "topic_id": assignments,
    }).to_csv(
        output_dir / "topic_assignments.csv",
        index=False,
    )

    keyword_rows = [
        {
            "topic_id": topic_id,
            "rank": rank + 1,
            "keyword": word,
        }
        for topic_id, words in enumerate(topics)
        for rank, word in enumerate(words)
    ]

    pd.DataFrame(keyword_rows).to_csv(
        output_dir / "topic_keywords.csv",
        index=False,
    )

    with (
        output_dir / "metrics.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
    )
    args = parser.parse_args()

    if args.repeats < 2:
        raise ValueError("At least 2 repeats required")

    output_paths = [
        OUTPUT_DIR / "runtime_by_run.csv",
        OUTPUT_DIR / "runtime_summary.csv",
        OUTPUT_DIR / "runtime_environment.json",
    ]

    for path in output_paths:
        if path.exists():
            raise FileExistsError(
                f"Output already exists: {path}"
            )

    config, base = load_config(CONFIG_PATH)

    if (
        config["selection"]["approval_status"]
        != "approved"
    ):
        raise ValueError("LDA is not approved")

    k = int(
        config["selection"]["selected_n_topics"]
    )
    seed = int(
        config["model"]["random_state"]
    )

    if k != 15 or seed != 42:
        raise ValueError("Expected K=15, seed=42")

    data_cfg = config["dataset"]

    if data_cfg["data_version"] != "v1.0":
        raise ValueError("Invalid dataset version")

    if (
        data_cfg["preprocessing_variant"]
        != "p2_segmented"
    ):
        raise ValueError("Invalid preprocessing")

    lda_cfg = copy.deepcopy(base["lda"])
    lda_cfg["n_components"] = k
    lda_cfg["random_state"] = seed

    rows = []

    for scenario, fit_split, predict_split in SCENARIOS:

        for repeat in range(1, args.repeats + 1):

            print(
                f"\n{scenario}: {repeat}/{args.repeats}",
                flush=True,
            )

            pipeline_start = time.perf_counter()

            inputs, data_loading_s = timed(
                load_inputs,
                data_cfg,
                fit_split,
                predict_split,
            )

            (
                train_texts,
                evaluation_texts,
                gold,
                n_train,
                n_eval,
            ) = inputs

            (vectorizer, train_bow), bow_fit_s = timed(
                fit_bow_pipeline,
                train_texts,
                base["vectorizer"],
            )

            evaluation_bow, bow_transform_s = timed(
                transform_bow_pipeline,
                vectorizer,
                evaluation_texts,
            )

            representation_s = (
                bow_fit_s + bow_transform_s
            )

            model, fit_s = timed(
                fit_lda_pipeline,
                train_bow,
                lda_cfg,
            )

            distributions, predict_s = timed(
                transform_lda_pipeline,
                model,
                evaluation_bow,
            )

            if distributions.shape != (n_eval, k):
                raise ValueError(
                    "Unexpected distribution shape"
                )

            if not np.isfinite(distributions).all():
                raise ValueError(
                    "Invalid topic probabilities"
                )

            evaluation_result, evaluation_s = timed(
                evaluate_metrics,
                distributions,
                model,
                vectorizer,
                evaluation_bow,
                gold,
                int(config["model"]["top_n_words"]),
            )

            topics, assignments, metrics = (
                evaluation_result
            )

            # Save outputs to a temporary folder.
            # Do not overwrite approved Final artifacts.
            write_start = time.perf_counter()

            with tempfile.TemporaryDirectory(
                prefix="lda_runtime_"
            ) as temp_dir:

                save_outputs(
                    Path(temp_dir),
                    vectorizer,
                    model,
                    distributions,
                    assignments,
                    topics,
                    metrics,
                )

            artifact_write_s = (
                time.perf_counter() - write_start
            )

            runtime_s = (
                time.perf_counter() - pipeline_start
            )

            measurements = {
                "data_loading_seconds": data_loading_s,
                "representation_seconds": representation_s,
                "fit_seconds": fit_s,
                "predict_seconds": predict_s,
                "evaluation_seconds": evaluation_s,
                "artifact_write_seconds": artifact_write_s,
                "runtime_seconds": runtime_s,
            }

            for stage, seconds in measurements.items():
                rows.append({
                    "run": (
                        f"lda_k15_seed42_"
                        f"{scenario}_end_to_end"
                    ),
                    "fit_split": fit_split,
                    "inference_split": predict_split,
                    "repeat": repeat,
                    "n_training_documents": n_train,
                    "n_inference_documents": n_eval,
                    "n_topics": k,
                    "seed": seed,
                    "stage": stage,
                    "seconds": round(seconds, 6),
                })

            print(
                f"Total: {runtime_s:.4f}s | "
                f"Data: {data_loading_s:.4f}s | "
                f"Fit: {fit_s:.4f}s | "
                f"Write: {artifact_write_s:.4f}s",
                flush=True,
            )

    df = pd.DataFrame(rows)

    group_columns = [
        "run",
        "fit_split",
        "inference_split",
        "n_training_documents",
        "n_inference_documents",
        "n_topics",
        "seed",
        "stage",
    ]

    summary = (
        df.groupby(group_columns, sort=False)["seconds"]
        .agg(
            mean_seconds="mean",
            std_seconds="std",
            min_seconds="min",
            max_seconds="max",
            repeats="count",
        )
        .reset_index()
    )

    for column in (
        "mean_seconds",
        "std_seconds",
        "min_seconds",
        "max_seconds",
    ):
        summary[column] = summary[column].round(4)

    environment = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "benchmark_type": "extended_end_to_end",
        "dataset_version": "v1.0",
        "preprocessing_variant": "p2_segmented",
        "n_topics": k,
        "random_state": seed,
        "repeats": args.repeats,
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "runtime_scope": {
            "included": [
                "dataset_loading",
                "text_extraction",
                "bow_fit",
                "bow_transform",
                "lda_fit",
                "lda_prediction",
                "npmi_diversity_ari_nmi",
                "model_and_metrics_serialization",
            ],
            "excluded": [
                "common_artifact_projection_2d",
                "representative_docs_export",
                "benchmark_summary_write",
            ],
            "note": (
                "New benchmark fit, not the original "
                "Final training run. Temporary output "
                "serialization is measured, not full "
                "common adapter export."
            ),
        },
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_paths[0],
        index=False,
    )

    summary.to_csv(
        output_paths[1],
        index=False,
    )

    with output_paths[2].open(
        "x",
        encoding="utf-8",
    ) as f:
        json.dump(
            environment,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n=== END-TO-END RUNTIME SUMMARY ===")

    print(
        summary.loc[
            summary["stage"] == "runtime_seconds",
            [
                "run",
                "mean_seconds",
                "std_seconds",
                "repeats",
            ],
        ].to_string(index=False)
    )

    print(
        f"\nResults saved to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
