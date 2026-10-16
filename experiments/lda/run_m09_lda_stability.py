
"""M09 - LDA stability evaluation across multiple random seeds.

Pipeline:
1. Load Dataset v1 Train/Validation using M07 utilities.
2. Fit BoW on Train only.
3. Train LDA models with fixed K and different random seeds.
4. Match topics using Hungarian algorithm and keyword Jaccard.
5. Calculate document-topic cosine stability on Validation.
6. Evaluate reference seed and all unique seed pairs.
7. Export reproducible CSV results.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

from src.data.dataset_loader import load_split_dataset
from src.features.article_text import build_article_text
from src.features.bow import (
    fit_bow_pipeline,
    transform_bow_pipeline,
)
from src.models.lda_pipeline import (
    fit_lda_pipeline,
    transform_lda_pipeline,
)
from src.evaluation.lda_metrics import extract_topic_words
from src.evaluation.lda_stability import (
    align_document_topics,
    document_topic_stability,
    match_topics,
)


DEFAULT_CONFIG = "configs/lda_stability.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="M09 - LDA Stability Experiments"
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to M09 YAML configuration",
    )
    return parser.parse_args()


def load_yaml(path):
    """Load and validate a YAML configuration file."""
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML config: {path}")

    return config


def validate_config(config, base_config):
    """Validate experiment settings and M07 compatibility."""
    dataset = config["dataset"]
    base_dataset = base_config["dataset"]
    settings = config["stability"]
    evaluation = config["evaluation"]

    k = int(settings["n_topics"])
    top_n = int(settings["top_n_words"])
    seeds = list(settings["random_seeds"])
    reference_seed = settings["reference_seed"]

    if k < 1:
        raise ValueError("n_topics must be positive")

    if top_n < 1:
        raise ValueError("top_n_words must be positive")

    if len(seeds) < 2:
        raise ValueError("At least two random seeds are required")

    if any(type(seed) is not int for seed in seeds):
        raise ValueError("Random seeds must be integers")

    if len(set(seeds)) != len(seeds):
        raise ValueError("Random seeds must be unique")

    if reference_seed not in seeds:
        raise ValueError(
            "reference_seed must be included in random_seeds"
        )

    for key in (
        "path",
        "split_manifest_path",
        "text_variant",
        "train_split",
        "val_split",
    ):
        if dataset[key] != base_dataset[key]:
            raise ValueError(
                f"M09 and M07 dataset settings differ: {key}"
            )

    if evaluation["evaluation_split"] != dataset["val_split"]:
        raise ValueError("Evaluation must use Validation")

    if evaluation["topic_matching"] != "hungarian":
        raise ValueError("Unsupported topic matching method")

    if evaluation["keyword_similarity"] != "jaccard":
        raise ValueError("Unsupported keyword similarity")

    if not evaluation["document_topic_stability"]:
        raise ValueError(
            "document_topic_stability must be enabled"
        )

    return k, top_n, seeds, reference_seed


def load_data_and_bow(config, base_config):
    """Load Train/Validation and fit BoW exclusively on Train."""
    dataset = config["dataset"]

    train_df = load_split_dataset(
        dataset["path"],
        dataset["split_manifest_path"],
        dataset["train_split"],
    )

    val_df = load_split_dataset(
        dataset["path"],
        dataset["split_manifest_path"],
        dataset["val_split"],
    )

    train_ids = set(train_df["article_id"].astype(str))
    val_ids = set(val_df["article_id"].astype(str))

    if train_ids & val_ids:
        raise ValueError("Train/Validation article_id leakage")

    text_variant = dataset["text_variant"]

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
        base_config["vectorizer"],
    )

    val_bow = transform_bow_pipeline(
        vectorizer,
        val_texts,
    )

    vocabulary = vectorizer.get_feature_names_out()

    print(f"Vocabulary size: {len(vocabulary)}")

    return train_bow, val_bow, vocabulary


def train_seed_models(
    train_bow,
    val_bow,
    vocabulary,
    base_config,
    k,
    seeds,
    top_n,
):
    """Train all seeds while keeping the same data and parameters."""
    results = {}
    runtime_rows = []

    for seed in seeds:
        print(f"\nTraining LDA: K={k}, seed={seed}")

        lda_config = dict(base_config["lda"])
        lda_config["n_components"] = k
        lda_config["random_state"] = seed

        start = perf_counter()

        model = fit_lda_pipeline(
            train_bow,
            lda_config,
        )

        val_distributions = transform_lda_pipeline(
            model,
            val_bow,
        )

        topic_words = extract_topic_words(
            model,
            vocabulary,
            top_n,
        )

        elapsed = perf_counter() - start

        results[seed] = {
            "topic_words": topic_words,
            "val_distributions": val_distributions,
        }

        runtime_rows.append({
            "seed": seed,
            "k": k,
            "runtime_seconds": round(elapsed, 2),
        })

        print(f"Completed in {elapsed:.2f}s")

    return results, runtime_rows


def evaluate_seed_pair(
    results,
    seed_a,
    seed_b,
):
    """Evaluate one seed pair after aligning its topics."""
    model_a = results[seed_a]
    model_b = results[seed_b]

    matches, mean_jaccard = match_topics(
        model_a["topic_words"],
        model_b["topic_words"],
    )

    aligned_b = align_document_topics(
        model_b["val_distributions"],
        matches,
    )

    mean_cosine = document_topic_stability(
        model_a["val_distributions"],
        aligned_b,
    )

    return matches, mean_jaccard, mean_cosine


def evaluate_reference(results, seeds, reference_seed, k):
    """Compare each model to the reference model."""
    metric_rows = []
    matching_rows = []

    print("\nEvaluating reference-seed stability...")

    for seed in seeds:
        matches, mean_jaccard, mean_cosine = (
            evaluate_seed_pair(
                results,
                reference_seed,
                seed,
            )
        )

        metric_rows.append({
            "reference_seed": reference_seed,
            "candidate_seed": seed,
            "k": k,
            "mean_matched_jaccard": mean_jaccard,
            "mean_document_cosine": mean_cosine,
        })

        for ref_topic, cand_topic, similarity in matches:
            matching_rows.append({
                "reference_seed": reference_seed,
                "candidate_seed": seed,
                "reference_topic": ref_topic,
                "candidate_topic": cand_topic,
                "keyword_jaccard": similarity,
            })

    return pd.DataFrame(metric_rows), pd.DataFrame(matching_rows)


def evaluate_all_pairs(results, seeds, k):
    """Evaluate all unordered pairs of different random seeds."""
    pairwise_rows = []

    print("\nEvaluating all seed pairs...")

    for seed_a, seed_b in combinations(seeds, 2):
        _, mean_jaccard, mean_cosine = (
            evaluate_seed_pair(
                results,
                seed_a,
                seed_b,
            )
        )

        pairwise_rows.append({
            "seed_a": seed_a,
            "seed_b": seed_b,
            "k": k,
            "mean_matched_jaccard": mean_jaccard,
            "mean_document_cosine": mean_cosine,
        })

    return pd.DataFrame(pairwise_rows)


def validate_results(
    reference_df,
    matches_df,
    pairwise_df,
    runtime_df,
    seeds,
    reference_seed,
    k,
):
    """Validate row counts and evaluation output integrity."""
    n_seeds = len(seeds)
    expected_pairs = n_seeds * (n_seeds - 1) // 2

    if len(reference_df) != n_seeds:
        raise ValueError("Unexpected reference metrics count")

    if len(matches_df) != n_seeds * k:
        raise ValueError("Unexpected topic matches count")

    if len(pairwise_df) != expected_pairs:
        raise ValueError("Unexpected pairwise metrics count")

    if len(runtime_df) != n_seeds:
        raise ValueError("Unexpected runtime count")

    metrics = [
        "mean_matched_jaccard",
        "mean_document_cosine",
    ]

    for df in (reference_df, pairwise_df):
        values = df[metrics].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError("Stability metrics contain NaN/Inf")

        if ((values < 0) | (values > 1 + 1e-8)).any():
            raise ValueError("Stability metrics outside [0,1]")

    reference_self = reference_df.loc[
        reference_df["candidate_seed"] == reference_seed
    ]

    if len(reference_self) != 1:
        raise ValueError("Missing reference self-comparison")

    self_scores = reference_self[metrics].to_numpy(dtype=float)

    if not np.allclose(self_scores, 1.0, atol=1e-8):
        raise ValueError("Reference self-comparison must equal 1")

    if pairwise_df[["seed_a", "seed_b"]].duplicated().any():
        raise ValueError("Duplicate seed pairs found")

    if (pairwise_df["seed_a"] == pairwise_df["seed_b"]).any():
        raise ValueError("Pairwise results contain self-comparison")

    print("M09 RESULT VALIDATION PASSED")


def save_results(
    config,
    reference_df,
    matches_df,
    runtime_df,
    pairwise_df,
):
    """Export all M09 results as UTF-8 CSV files."""
    output_dir = Path(config["outputs"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "stability_metrics.csv": reference_df,
        "topic_matches.csv": matches_df,
        "runtime_by_seed.csv": runtime_df,
        "pairwise_stability.csv": pairwise_df,
    }

    for filename, df in files.items():
        path = output_dir / filename
        df.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"Saved: {path}")


def main():
    args = parse_args()

    config = load_yaml(args.config)
    base_config = load_yaml(config["base_config"])

    k, top_n, seeds, reference_seed = validate_config(
        config,
        base_config,
    )

    print("M09 - LDA Stability Experiments")
    print(f"Topics (K): {k}")
    print(f"Random seeds: {seeds}")
    print(f"Reference seed: {reference_seed}")

    train_bow, val_bow, vocabulary = load_data_and_bow(
        config,
        base_config,
    )

    results, runtime_rows = train_seed_models(
        train_bow,
        val_bow,
        vocabulary,
        base_config,
        k,
        seeds,
        top_n,
    )

    reference_df, matches_df = evaluate_reference(
        results,
        seeds,
        reference_seed,
        k,
    )

    pairwise_df = evaluate_all_pairs(
        results,
        seeds,
        k,
    )

    runtime_df = pd.DataFrame(runtime_rows)

    validate_results(
        reference_df,
        matches_df,
        pairwise_df,
        runtime_df,
        seeds,
        reference_seed,
        k,
    )

    save_results(
        config,
        reference_df,
        matches_df,
        runtime_df,
        pairwise_df,
    )

    print("\nREFERENCE-SEED STABILITY")
    print(reference_df.to_string(index=False))

    print("\nALL-PAIR STABILITY")
    print(pairwise_df.to_string(index=False))

    pairwise_mean = pairwise_df[
        ["mean_matched_jaccard", "mean_document_cosine"]
    ].mean()

    print("\nALL-PAIR MEAN (EXCLUDING SELF-COMPARISONS)")
    print(pairwise_mean.to_string())

    print("\nM09 completed successfully.")


if __name__ == "__main__":
    main()
