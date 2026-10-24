"""Supplement M09 with saved assignments and between-run ARI on Validation.

Run separately from original M09 and Final outputs. Gensim is unnecessary:
NPMI uses the project's document_binary_common implementation.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn

from experiments.lda.run_m09_lda_stability import (
    evaluate_all_pairs, load_data_and_bow, load_yaml, validate_config,
)
from experiments.nmf.metrics import npmi_coherence
from src.data.dataset_loader import load_split_dataset
from src.evaluation.lda_metrics import extract_topic_words, topic_diversity
from src.evaluation.stability_summary import pairwise_assignment_ari
from src.models.lda_pipeline import fit_lda_pipeline, transform_lda_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/lda/e04_handoff")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing run: {output}")
    config = load_yaml("configs/lda_stability.yaml")
    base = load_yaml(config["base_config"])
    k, top_n, seeds, _ = validate_config(config, base)
    train, val, vocabulary = load_data_and_bow(config, base)
    documents = load_split_dataset(
        config["dataset"]["path"], config["dataset"]["split_manifest_path"], "val"
    )
    ids = documents.article_id.astype(str).to_numpy()
    if len(ids) != val.shape[0] or len(set(ids)) != len(ids):
        raise ValueError("Invalid Validation article_id alignment")
    results, per_seed, assignments, keyword_rows = {}, [], [], []
    for seed in seeds:
        effective = dict(base["lda"], n_components=k, random_state=seed, n_jobs=args.n_jobs)
        print(f"Training LDA K={k} seed={seed}", flush=True)
        start = perf_counter()
        model = fit_lda_pipeline(train, effective)
        distributions = transform_lda_pipeline(model, val)
        topics = extract_topic_words(model, vocabulary, top_n)
        feature_ids = np.argsort(-model.components_, axis=1)[:, :top_n]
        npmi, _ = npmi_coherence(val, feature_ids)
        assignments.append(pd.DataFrame({
            "seed": seed, "article_id": ids,
            "topic_id": distributions.argmax(axis=1),
        }))
        results[seed] = {"topic_words": topics, "val_distributions": distributions}
        per_seed.append({
            "method": "LDA", "seed": seed, "split": "val",
            "n_documents": len(ids), "topic_count": k,
            "coherence": npmi, "coherence_definition": "document_binary_common",
            "topic_diversity": topic_diversity(topics),
            "outlier_rate": 0.0, "coverage": 1.0,
            "runtime_seconds": perf_counter() - start,
        })
        for topic_id, words in enumerate(topics):
            for rank, word in enumerate(words, 1):
                keyword_rows.append({"seed": seed, "topic_id": topic_id, "rank": rank, "term": word})
        print(f"Completed seed {seed}: NPMI={npmi:.6f}", flush=True)
    all_assignments = pd.concat(assignments, ignore_index=True)
    pairs = pairwise_assignment_ari(all_assignments)
    old_style = evaluate_all_pairs(results, seeds, k).rename(columns={
        "mean_matched_jaccard": "keyword_jaccard",
        "mean_document_cosine": "document_cosine",
    })
    pairs = pairs.merge(old_style, on=["seed_a", "seed_b"], validate="one_to_one")
    pairs.insert(0, "method", "LDA")
    input_files = ["configs/lda.yaml", "configs/lda_stability.yaml",
                   config["dataset"]["path"], config["dataset"]["split_manifest_path"]]
    metadata = {
        "status": "supplementary_run_pending_review", "dataset_version": "v1.0",
        "fit_split": "train", "evaluation_split": "val", "test_used_for_fit_or_evaluation": False,
        "preprocessing": "p2_segmented", "n_topics": k, "seeds": seeds,
        "n_jobs": args.n_jobs, "python": platform.python_version(),
        "sklearn": sklearn.__version__, "numpy": np.__version__,
        "vectorizer": base["vectorizer"], "model": dict(base["lda"], n_components=k, n_jobs=args.n_jobs),
        "npmi_definition": "document_binary_common", "keyword_matching": "hungarian_top10_jaccard",
        "input_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in input_files},
    }
    output.mkdir(parents=True)
    for filename, table in {
        "per_seed.csv": pd.DataFrame(per_seed), "pairwise.csv": pairs,
        "validation_assignments.csv": all_assignments, "topic_keywords.csv": pd.DataFrame(keyword_rows),
    }.items():
        table.to_csv(output / filename, index=False, encoding="utf-8-sig")
    np.savez_compressed(output / "validation_distributions.npz", article_ids=ids.astype(str),
                        **{f"seed_{s}": results[s]["val_distributions"] for s in seeds})
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pairs.to_string(index=False))


if __name__ == "__main__":
    main()
