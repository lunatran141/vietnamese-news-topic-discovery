"""Evaluate selected NMF across seeds on Train/Validation, exporting E04 inputs.

TF-IDF is fitted once on Train. Test is neither transformed nor scored.
Existing output directories are never replaced.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from itertools import combinations
import json
import logging
import platform
import sys
from time import perf_counter
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_info, threadpool_limits

from experiments.nmf.run_experiment import (
    build_model, build_vectorizer, evaluate_model, load_dataset_v1,
    load_yaml, resolve_project_path, sha256_file,
)
from src.evaluation.lda_stability import (
    align_document_topics, document_topic_stability, match_topics,
)
from src.evaluation.stability_summary import pairwise_assignment_ari, summarize_metrics


def validate_settings(config, base):
    """Require distinct valid seeds, selected K and the E04 Train/Val splits."""
    settings = config["stability"]
    seeds = settings["random_seeds"]
    if len(seeds) < 3 or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("At least three integer seeds in [0, 2**32) are required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Random seeds must be unique")
    k, top_n = settings["n_topics"], settings["top_n_words"]
    if type(k) is not int or k < 2 or k != base["topic_selection"]["selected_n_topics"]:
        raise ValueError("n_topics must equal the selected NMF topic count")
    if type(top_n) is not int or top_n < 2:
        raise ValueError("top_n_words must be an integer >= 2")
    if settings["fit_split"] != "train" or settings["evaluation_split"] != "val":
        raise ValueError("Stability must fit Train and evaluate Validation")
    if type(settings["thread_limit"]) is not int or settings["thread_limit"] < 1:
        raise ValueError("thread_limit must be a positive integer")
    return k, top_n, seeds


def evaluate_pairs(results, assignments):
    """Pair all seeds; align soft shares by Hungarian top-word Jaccard."""
    pairs = pairwise_assignment_ari(assignments)
    pairs.insert(0, "method", "NMF")
    scores, matching_rows = [], []
    for seed_a, seed_b in combinations(sorted(results), 2):
        left, right = results[seed_a], results[seed_b]
        matches, jaccard = match_topics(left["topics"], right["topics"])
        aligned = align_document_topics(right["distributions"], matches)
        scores.append({
            "seed_a": seed_a, "seed_b": seed_b,
            "keyword_jaccard": jaccard,
            "document_cosine": document_topic_stability(left["distributions"], aligned),
        })
        for topic_a, topic_b, similarity in matches:
            matching_rows.append({
                "seed_a": seed_a, "seed_b": seed_b,
                "topic_a": topic_a, "topic_b": topic_b, "keyword_jaccard": similarity,
            })
    return (
        pairs.merge(pd.DataFrame(scores), on=["seed_a", "seed_b"], validate="one_to_one"),
        pd.DataFrame(matching_rows),
    )


def save_figure(per_seed, pairs, path, n_topics):
    """Export descriptive results for the report, without confidence intervals."""
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(per_seed.seed.astype(str), per_seed.coherence, "o-")
    axes[0].set(title="Validation NPMI by seed", xlabel="Seed", ylabel="NPMI")
    pair_names = [f"{a}-{b}" for a, b in zip(pairs.seed_a, pairs.seed_b)]
    axes[1].plot(pair_names, pairs.keyword_jaccard, "o-", color="#e69f00")
    axes[1].set(title="Matched topic keywords", ylabel="Mean Jaccard")
    axes[2].plot(pair_names, pairs.ari_between_runs, "o-", label="ARI between runs")
    axes[2].plot(pair_names, pairs.document_cosine, "s--", label="Aligned share cosine")
    axes[2].set(title="Validation assignments and shares", ylabel="Similarity")
    axes[2].legend(fontsize=8)
    axes[0].set_ylim(-1, 1.02)
    axes[1].set_ylim(0, 1.02)
    axes[2].set_ylim(-1, 1.02)
    for axis in axes:
        axis.grid(alpha=0.25)
    for axis in axes[1:]:
        axis.set_xlabel("Seed pair")
        axis.tick_params(axis="x", labelrotation=60, labelsize=8)
    figure.suptitle(f"NMF stability | K={n_topics} | fit: Train | evaluation: Validation")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_stability(config_path="configs/nmf_stability.yaml", output_override=None):
    config_path = resolve_project_path(config_path)
    config = load_yaml(config_path)
    base_path = resolve_project_path(config["base_config"])
    base = load_yaml(base_path)
    k, top_n, seeds = validate_settings(config, base)
    output = resolve_project_path(output_override or config["outputs"]["directory"])
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing run: {output}")
    dataset, manifest = load_dataset_v1(base)
    train = dataset.loc[dataset["split"].eq("train")].reset_index(drop=True)
    val = dataset.loc[dataset["split"].eq("val")].reset_index(drop=True)
    if train.empty or val.empty:
        raise ValueError("Train and Validation must both be nonempty")
    text_column, id_column = base["data"]["text_column"], base["data"]["id_column"]
    ids = val[id_column].astype(str).to_numpy()
    output.mkdir(parents=True)
    logger = logging.getLogger("nmf_stability")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(output / "run_log.txt", encoding="utf-8")]
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
    metadata = {
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": manifest["dataset_version"],
        "fit_split": "train", "evaluation_split": "val",
        "test_used_for_fit_or_evaluation": False,
        "preprocessing": base["data"]["preprocessing_variant"],
        "text_column": text_column, "n_topics": k, "top_n_words": top_n, "seeds": seeds,
        "n_train": len(train), "n_validation": len(val),
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "pandas": pd.__version__,
        "representation": base["representation"],
        "model": {key: value for key, value in base["model"].items() if key != "random_seed"},
        "seed_parameter": "NMF.random_state; NNDSVD initialization uses randomized SVD in the installed sklearn implementation",
        "fixed_steps": "P3 text, article_id splits, row order, Train TF-IDF vocabulary/IDF, K and all non-seed NMF parameters",
        "seed_selection": "none; seed 42 remains the reported Final model",
        "npmi_definition": "document_binary_common",
        "keyword_matching": f"hungarian_top{top_n}_jaccard",
        "document_cosine": "mean per-document cosine after the same keyword-based topic permutation; zero vector contributes 0",
        "ari_definition": "pairwise Validation primary-topic assignments, without category_gold",
        "unassigned_policy": "zero topic-weight rows receive topic_id=-1",
        "standard_deviation": "sample, ddof=1; descriptive only; seed pairs are dependent",
        "thread_limit": config["stability"]["thread_limit"],
    }
    input_paths = [config_path, base_path] + [resolve_project_path(base["data"][key]) for key in (
        "dataset_path", "dataset_manifest_path", "split_manifest_path",
    )]
    source_paths = [resolve_project_path(path) for path in (
        "experiments/nmf/run_stability.py", "experiments/nmf/run_experiment.py",
        "experiments/nmf/metrics.py", "src/models/nmf.py",
        "src/evaluation/lda_stability.py", "src/evaluation/stability_summary.py",
    )]
    metadata["input_sha256"] = {str(path): sha256_file(path) for path in input_paths}
    metadata["source_sha256"] = {str(path): sha256_file(path) for path in source_paths}
    (output / "config_used.json").write_text(
        json.dumps({"stability": config, "base": base}, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    metadata_path = output / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    try:
        with threadpool_limits(limits=config["stability"]["thread_limit"]):
            metadata["threadpools"] = threadpool_info()
            logger.info("Fit TF-IDF once on %s Train articles; evaluate %s Validation articles", len(train), len(val))
            start = perf_counter()
            vectorizer = build_vectorizer(base)
            train_matrix = vectorizer.fit_transform(train[text_column].astype(str))
            val_matrix = vectorizer.transform(val[text_column].astype(str))
            features = vectorizer.get_feature_names_out()
            if top_n > len(features):
                raise ValueError("top_n_words exceeds the Train vocabulary size")
            metadata["representation_seconds"] = perf_counter() - start
            metadata["n_features"] = len(features)
            (output / "feature_names.json").write_text(json.dumps(features.tolist(), ensure_ascii=False), encoding="utf-8")
            results, seed_rows, assignments, keyword_rows = {}, [], [], []
            for seed in seeds:
                effective = deepcopy(base)
                effective["model"]["random_seed"] = seed
                model = build_model(effective, k)
                logger.info("Training NMF K=%s init=%s seed=%s", k, model.config.init, seed)
                start = perf_counter()
                with warnings.catch_warnings(record=True) as recorded:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(train_matrix, features)
                    fit_seconds = perf_counter() - start
                    start_transform = perf_counter()
                    raw_weights = model.estimator_.transform(val_matrix)
                    transform_seconds = perf_counter() - start_transform
                    metrics, _, distributions = evaluate_model(model, val_matrix, features, top_n, raw_weights)
                zero_rows = distributions.sum(axis=1) == 0
                labels = distributions.argmax(axis=1)
                labels[zero_rows] = -1
                topics = [[term for term, _ in model.top_terms(topic_id, top_n)] for topic_id in range(k)]
                results[seed] = {"topics": topics, "distributions": distributions}
                assignments.append(pd.DataFrame({"seed": seed, "article_id": ids, "topic_id": labels}))
                warning_messages = [f"{w.category.__name__}: {w.message}" for w in recorded]
                seed_rows.append({
                    "method": "NMF", "seed": seed, "split": "val", "n_documents": len(ids),
                    "topic_count": k, "coherence": metrics["npmi_coherence"],
                    "coherence_definition": "document_binary_common", "topic_diversity": metrics["topic_diversity"],
                    "outlier_rate": float(zero_rows.mean()), "coverage": float((~zero_rows).mean()),
                    "normalized_reconstruction_error": metrics["normalized_reconstruction_error"],
                    "mean_primary_topic_share": metrics["mean_primary_topic_share"],
                    "n_iter": metrics["n_iter"], "converged": metrics["converged"],
                    "convergence_warning": any(issubclass(w.category, ConvergenceWarning) for w in recorded),
                    "n_zero_topic_rows": int(zero_rows.sum()),
                    "fit_seconds": fit_seconds, "validation_transform_seconds": transform_seconds,
                    "runtime_seconds": perf_counter() - start, "warnings": " | ".join(warning_messages),
                })
                for topic_id in range(k):
                    for rank, (term, weight) in enumerate(model.top_terms(topic_id, top_n), 1):
                        keyword_rows.append({"seed": seed, "topic_id": topic_id, "rank": rank, "term": term, "weight": weight})
                logger.info("Completed seed %s: NPMI=%.9f diversity=%.6f n_iter=%s warnings=%s", seed,
                            metrics["npmi_coherence"], metrics["topic_diversity"], metrics["n_iter"], len(recorded))
                for message in warning_messages:
                    logger.warning(message)
            per_seed = pd.DataFrame(seed_rows)
            all_assignments = pd.concat(assignments, ignore_index=True)
            pairs, matches = evaluate_pairs(results, all_assignments)
            summary = pd.concat([
                summarize_metrics(per_seed, ["coherence", "topic_diversity", "normalized_reconstruction_error", "n_iter", "coverage"], "seed"),
                summarize_metrics(pairs, ["ari_between_runs", "keyword_jaccard", "document_cosine"], "seed_pair"),
            ], ignore_index=True)
            for name, table in {
                "per_seed.csv": per_seed, "pairwise.csv": pairs, "summary.csv": summary,
                "validation_assignments.csv": all_assignments,
                "topic_keywords.csv": pd.DataFrame(keyword_rows), "topic_matches.csv": matches,
            }.items():
                table.to_csv(output / name, index=False, encoding="utf-8-sig")
            np.savez_compressed(output / "validation_distributions.npz", article_ids=ids.astype(str),
                                **{f"seed_{seed}": results[seed]["distributions"] for seed in seeds})
            save_figure(per_seed, pairs, output / "stability.png", k)
        metadata.update(status="completed", finished_at_utc=datetime.now(timezone.utc).isoformat(),
                        n_seed_pairs=len(pairs), all_fit_converged=bool(per_seed.converged.all()),
                        any_convergence_warning=bool(per_seed.convergence_warning.any()))
        metadata["output_sha256"] = {p.name: sha256_file(p) for p in sorted(output.iterdir())
                                     if p.is_file() and p.name not in {"run_metadata.json", "run_log.txt"}}
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info("Completed %s seeds / %s distinct pairs. Results: %s", len(seeds), len(pairs), output)
        logger.info("\n%s", summary.to_string(index=False))
        return output
    except Exception as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        raise
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/nmf_stability.yaml")
    parser.add_argument("--output", help="Use a new directory to keep the existing run")
    args = parser.parse_args()
    run_stability(args.config, args.output)


if __name__ == "__main__":
    main()
