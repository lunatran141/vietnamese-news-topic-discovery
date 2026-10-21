
"""Reproducible LDA runtime benchmark.

Measures an explicitly defined core pipeline, not the original Final run.
Does not change Final artifacts, test metrics, or the Test lock.
Run from the repository root:
    python -m experiments.lda.benchmark_lda_runtime
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from experiments.lda.run_lda_final import (
    load_config,
    prepare,
    validate_splits,
)
from src.data.dataset_loader import load_split_dataset
from src.features.article_text import build_article_text
from src.features.bow import fit_bow_pipeline, transform_bow_pipeline
from src.models.lda_pipeline import fit_lda_pipeline, transform_lda_pipeline
from src.evaluation.lda_metrics import extract_topic_words, topic_diversity
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from experiments.nmf.metrics import npmi_coherence

CONFIG_PATH = "configs/lda_final.yaml"
FINAL_PATH = Path("artifacts/lda/final/lda_pipeline.joblib")
OUTPUT_DIR = Path("results/lda/runtime")

REQUIRED_REPEATS = 5


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def describe(values):
    values = [float(v) for v in values]
    return {
        "mean_seconds": round(statistics.mean(values), 4),
        "std_seconds": round(statistics.stdev(values), 4)
        if len(values) > 1 else 0.0,
        "min_seconds": round(min(values), 4),
        "max_seconds": round(max(values), 4),
    }


def evaluate_lda_metrics(
    distribution,
    keywords,
    vectorizer,
    evaluation_bow,
    gold_labels,
):
    """Evaluate LDA without repeating topic inference."""
    assignments = distribution.argmax(axis=1)

    if len(assignments) != len(gold_labels):
        raise ValueError("Gold labels and assignments length mismatch")

    if pd.isna(gold_labels).any():
        raise ValueError("Missing gold labels")

    word_ids = [
        [vectorizer.vocabulary_[word] for word in topic_words]
        for topic_words in keywords
    ]

    npmi, per_topic_npmi = npmi_coherence(
        evaluation_bow,
        word_ids,
    )

    diversity = topic_diversity(keywords)

    ari = adjusted_rand_score(gold_labels, assignments)
    nmi = normalized_mutual_info_score(gold_labels, assignments)

    return {
        "npmi_coherence": float(npmi),
        "topic_diversity": float(diversity),
        "adjusted_rand_index": float(ari),
        "normalized_mutual_info": float(nmi),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=REQUIRED_REPEATS)
    args = parser.parse_args()

    if args.repeats < 2:
        raise ValueError("Benchmark needs at least 2 repetitions")

    config, base = load_config(CONFIG_PATH)
    k = int(config["selection"]["selected_n_topics"])
    seed = int(config["model"]["random_state"])

    if config["selection"]["approval_status"] != "approved":
        raise ValueError("Final selection has not been approved")
    if k != 15 or seed != 42:
        raise ValueError("Expected K=15 and seed=42")

    # Keep the existing Test lock unchanged.
    train, val = prepare(config, base)
    data_cfg = config["dataset"]

    test = load_split_dataset(
        data_cfg["path"],
        data_cfg["split_manifest_path"],
        "test",
    )
    validate_splits(train, val, test)

    development = pd.concat([train, val], ignore_index=True)
    all_data = pd.concat(
        [train, val, test], ignore_index=True
    )

    datasets = {
        "train": train,
        "validation": val,
        "train+val": development,
        "test": test,
        "all": all_data,
    }

    texts = {
        name: build_article_text(
            frame, data_cfg["text_variant"]
        ).tolist()
        for name, frame in datasets.items()
    }

    # The primary inference split defines the core pipeline.
    scenarios = [
        {
            "run": "lda_k15_seed42_development_benchmark",
            "fit_split": "train",
            "primary_inference_split": "validation",
            "auxiliary_splits": ["train"],
        },
        {
            "run": "lda_k15_seed42_final_benchmark",
            "fit_split": "train+val",
            "primary_inference_split": "test",
            "auxiliary_splits": [
                "train", "validation", "train+val", "all"
            ],
        },
    ]

    # Do not overwrite benchmark files from a previous run.
    paths = [
        OUTPUT_DIR / "runtime_by_run.csv",
        OUTPUT_DIR / "runtime_summary.csv",
        OUTPUT_DIR / "runtime_environment.json",
    ]
    for path in paths:
        if path.exists():
            raise FileExistsError(
                f"Benchmark output already exists: {path}"
            )

    lda_cfg = copy.deepcopy(base["lda"])
    lda_cfg["n_components"] = k
    lda_cfg["random_state"] = seed

    rows = []

    for scenario in scenarios:
        run_name = scenario["run"]
        fit_split = scenario["fit_split"]
        primary_split = scenario["primary_inference_split"]

        for repeat in range(1, args.repeats + 1):
            print(
                f"\n{run_name} - repeat {repeat}/{args.repeats}",
                flush=True,
            )

            # Benchmark the pipeline using the agreed group fields.
            pipeline_start = time.perf_counter()

            (vectorizer, fit_bow), bow_fit_s = timed(
                fit_bow_pipeline,
                texts[fit_split],
                base["vectorizer"],
            )

            primary_bow, bow_transform_s = timed(
                transform_bow_pipeline,
                vectorizer,
                texts[primary_split],
            )

            representation_s = bow_fit_s + bow_transform_s

            model, training_s = timed(
                fit_lda_pipeline,
                fit_bow,
                lda_cfg,
            )

            distribution, inference_s = timed(
                transform_lda_pipeline,
                model,
                primary_bow,
            )

            keywords, keyword_s = timed(
                extract_topic_words,
                model,
                vectorizer.get_feature_names_out(),
                int(config["model"]["top_n_words"]),
            )

            if distribution.shape != (
                len(datasets[primary_split]), k
            ):
                raise ValueError("Unexpected primary inference shape")

            if not np.isfinite(distribution).all():
                raise ValueError("Invalid topic distribution")

            gold = datasets[primary_split]["category_gold"]

            metrics, metric_s = timed(
                evaluate_lda_metrics,
                distribution,
                keywords,
                vectorizer,
                primary_bow,
                gold,
            )

            # Include topic keyword extraction as evaluation overhead.
            evaluation_s = keyword_s + metric_s

            runtime_s = time.perf_counter() - pipeline_start

            diversity = metrics["topic_diversity"]
            core_runtime_s = runtime_s

            common = {
                "run": run_name,
                "repeat": repeat,
                "fit_split": fit_split,
                "n_training_documents": len(datasets[fit_split]),
                "n_topics": k,
                "seed": seed,
            }

            stages = {
                "representation_seconds": representation_s,
                "fit_seconds": training_s,
                "predict_seconds": inference_s,
                "evaluation_seconds": evaluation_s,
                "runtime_seconds": runtime_s,
                "bow_fit": bow_fit_s,
                "bow_transform": bow_transform_s,
                "lda_training": training_s,
                "lda_inference": inference_s,
                "topic_keywords": keyword_s,
                "evaluation_metrics": metric_s,
                "core_pipeline": runtime_s,
            }


            for stage, seconds in stages.items():
                rows.append({
                    **common,
                    "inference_split": primary_split,
                    "n_inference_documents": len(
                        datasets[primary_split]
                    ),
                    "stage": stage,
                    "seconds": round(seconds, 6),
                    "topic_diversity_value": (
                        float(diversity)
                        if stage == "topic_diversity" else None
                    ),
                })

            # Auxiliary inference benchmarks are separate:
            # not added to core_pipeline.
            for split in scenario["auxiliary_splits"]:
                bow, transform_s = timed(
                    transform_bow_pipeline,
                    vectorizer,
                    texts[split],
                )

                aux_distribution, infer_s = timed(
                    transform_lda_pipeline,
                    model,
                    bow,
                )

                if aux_distribution.shape != (
                    len(datasets[split]), k
                ):
                    raise ValueError(
                        f"Unexpected inference shape: {split}"
                    )

                for stage, seconds in (
                    ("aux_bow_transform", transform_s),
                    ("aux_inference", infer_s),
                ):
                    rows.append({
                        **common,
                        "inference_split": split,
                        "n_inference_documents": len(datasets[split]),
                        "stage": stage,
                        "seconds": round(seconds, 6),
                        "topic_diversity_value": None,
                    })

            print(
                f"Core pipeline: {core_runtime_s:.4f}s "
                f"| Training: {training_s:.4f}s "
                f"| Inference: {inference_s:.4f}s",
                flush=True,
            )

    # Extra benchmark: inference with the existing Final model.
    # Does not retrain or change this model.
    if not FINAL_PATH.is_file():
        raise FileNotFoundError(FINAL_PATH)

    saved = joblib.load(FINAL_PATH)
    final_model = saved["lda_model"]
    final_vectorizer = saved["vectorizer"]

    if int(final_model.n_components) != k:
        raise ValueError("Saved Final model topic count mismatch")

    for split in ("train", "validation", "test", "all"):
        bow = transform_bow_pipeline(
            final_vectorizer, texts[split]
        )

        for repeat in range(1, args.repeats + 1):
            distributions, seconds = timed(
                transform_lda_pipeline,
                final_model,
                bow,
            )

            if distributions.shape != (len(datasets[split]), k):
                raise ValueError("Saved model inference shape mismatch")

            rows.append({
                "run": "lda_k15_seed42_saved_final",
                "repeat": repeat,
                "fit_split": "train+val",
                "n_training_documents": len(development),
                "n_topics": k,
                "seed": seed,
                "inference_split": split,
                "n_inference_documents": len(datasets[split]),
                "stage": "saved_final_inference",
                "seconds": round(seconds, 6),
                "topic_diversity_value": None,
            })

    df = pd.DataFrame(rows)

    keys = [
        "run", "fit_split", "inference_split",
        "n_training_documents", "n_inference_documents",
        "n_topics", "seed", "stage",
    ]

    summaries = []
    for key, group in df.groupby(keys, sort=False):
        summary = dict(zip(keys, key))
        summary.update(describe(group["seconds"]))
        summary["repeats"] = len(group)

        if summary["stage"] in (
            "lda_inference",
            "aux_inference",
            "saved_final_inference",
        ):
            mean_s = summary["mean_seconds"]
            summary["documents_per_second"] = (
                round(summary["n_inference_documents"] / mean_s, 2)
                if mean_s > 0 else None
            )
        else:
            summary["documents_per_second"] = None

        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)

    environment = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "benchmark_type": "separate_runtime_benchmark",
        "dataset_version": "v1.0",
        "preprocessing_variant": "p2_segmented",
        "n_topics": k,
        "seed": seed,
        "repeats": args.repeats,
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "cpu": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "joblib_version": joblib.__version__,
        "runtime_scope": {
            "runtime_seconds": (
                "Wall-clock time from BoW vectorization through "
                "LDA fitting, prediction and evaluation"
            ),
            "representation_seconds": [
                "bow_fit",
                "bow_transform",
            ],
            "fit_seconds": "lda_training",
            "predict_seconds": "lda_inference",
            "evaluation_seconds": [
                "topic_keyword_extraction",
                "document_binary_npmi",
                "topic_diversity",
                "adjusted_rand_index",
                "normalized_mutual_info",
            ],
            "excluded": [
                "CSV loading",
                "text extraction",
                "artifact serialization",
                "auxiliary inference",
            ],
            "notes": [
                "Each training repetition is a fresh fit.",
                "Development evaluates Validation.",
                "Final evaluates Test using the fixed approved configuration.",
                "The benchmark does not overwrite Final artifacts.",
                "Benchmark times are not historical Final run times.",
            ],
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Exclusive creation: never silently replace saved measurements.
    with paths[0].open("x", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)

    with paths[1].open("x", encoding="utf-8", newline="") as f:
        summary_df.to_csv(f, index=False)

    with paths[2].open("x", encoding="utf-8") as f:
        json.dump(environment, f, ensure_ascii=False, indent=2)

    print("\n=== CORE PIPELINE SUMMARY ===")
    print(
        summary_df[
            summary_df["stage"] == "runtime_seconds"
        ][
            ["run", "mean_seconds", "std_seconds",
             "min_seconds", "max_seconds", "repeats"]
        ].to_string(index=False)
    )
    print(f"\nSaved results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
