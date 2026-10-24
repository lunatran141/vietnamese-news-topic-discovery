"""Build descriptive E04 tables with explicit missing submissions and provenance."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluation.stability_summary import pairwise_assignment_ari, summarize_metrics


def main():
    config = yaml.safe_load(Path("configs/e04_stability.yaml").read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    split = config["evaluation_split"]
    split_ids = pd.read_csv("data/splits/split_manifest.csv", dtype={"article_id": str})
    expected_ids = set(split_ids.loc[split_ids["split"].eq(split), "article_id"])
    if not expected_ids:
        raise ValueError("Empty evaluation split")
    runs, pairs, status = [], [], []
    sources = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
               ["configs/e04_stability.yaml", "data/splits/split_manifest.csv"]}

    def read(path):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_csv(path, encoding="utf-8-sig")

    for method, source in config["sources"].items():
        directory = Path(source["directory"])
        if source["kind"] == "hdbscan":
            required = [directory / name for name in [
                "validation_assignments.csv", "validation_seed_metrics.csv", "train_seed_metrics.csv",
            ]]
        else:
            required = [directory / "per_seed.csv", directory / "validation_assignments.csv",
                        directory / "run_metadata.json"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            status.append({"method": method, "status": "missing_submission", "n_seeds": 0,
                           "seeds": "", "missing": "; ".join(missing)})
            continue
        if source["kind"] == "hdbscan":
            assignments = read(required[0]).rename(columns={"umap_seed": "seed", "cluster_id": "topic_id"})
            validation = read(required[1]).set_index("umap_seed")
            trained = read(required[2]).set_index("umap_seed")
            seed_rows = []
            for seed, group in assignments.groupby("seed"):
                row = validation.loc[seed]
                if row["n_articles"] != len(group) or row["n_noise"] != group.topic_id.eq(-1).sum():
                    raise ValueError("HDBSCAN assignments disagree with validation metrics")
                seed_rows.append({
                    "method": method, "seed": seed, "split": split, "n_documents": len(group),
                    "topic_count": trained.loc[seed, "n_clusters"],
                    "coherence": np.nan, "coherence_definition": "not_available",
                    "topic_diversity": np.nan, "outlier_rate": group.topic_id.eq(-1).mean(),
                    "coverage": group.topic_id.ne(-1).mean(),
                })
            table = pd.DataFrame(seed_rows)
        else:
            table = read(required[0])
            assignments = read(required[1])
            protocol = json.loads(required[2].read_text(encoding="utf-8-sig"))
            sources[str(required[2])] = hashlib.sha256(required[2].read_bytes()).hexdigest()
            if protocol.get("fit_split") != "train" or protocol.get("evaluation_split") != split:
                raise ValueError(f"Unsupported training/evaluation protocol for {method}")
            if protocol.get("dataset_version") != "v1.0":
                raise ValueError(f"Dataset version mismatch for {method}")
            if set(protocol.get("seeds", [])) != set(table.seed):
                raise ValueError(f"Metadata seeds mismatch for {method}")
        required_columns = {"method", "seed", "split", "n_documents", "topic_count",
                            "coherence", "coherence_definition", "topic_diversity", "outlier_rate", "coverage"}
        if not required_columns.issubset(table.columns):
            raise ValueError(f"Missing per-seed columns for {method}")
        if table.seed.isna().any() or table.seed.duplicated().any() or len(table) < 3:
            raise ValueError(f"Need at least three distinct seeds for {method}")
        if not table.method.eq(method).all() or not table["split"].eq(split).all():
            raise ValueError(f"Method/split mismatch for {method}")
        if not table.n_documents.eq(len(expected_ids)).all():
            raise ValueError(f"Unexpected document count for {method}")
        for column in ["seed", "n_documents", "topic_count", "outlier_rate", "coverage"]:
            if not np.isfinite(table[column].to_numpy(dtype=float)).all():
                raise ValueError(f"Missing/non-finite {method}/{column}")
        if (table.topic_count < 0).any() or not table.topic_count.eq(np.floor(table.topic_count)).all():
            raise ValueError(f"Invalid topic count for {method}")
        if not table.coherence.dropna().between(-1, 1).all():
            raise ValueError(f"Invalid coherence value for {method}")
        if set(table.seed) != set(assignments.seed):
            raise ValueError(f"Metrics/assignments seed mismatch for {method}")
        assignments.article_id = assignments.article_id.astype(str)
        for seed, group in assignments.groupby("seed"):
            if set(group.article_id) != expected_ids:
                raise ValueError(f"Wrong article_id set for {method}/{seed}")
            reported = table.loc[table.seed.eq(seed)].iloc[0]
            if not np.isclose(reported.outlier_rate, group.topic_id.eq(-1).mean()):
                raise ValueError(f"Incorrect outlier rate for {method}/{seed}")
            active_topics = group.loc[group.topic_id.ne(-1), "topic_id"].nunique()
            if active_topics > reported.topic_count:
                raise ValueError(f"More assigned topics than model topics for {method}/{seed}")
        for column in ["coverage", "outlier_rate", "topic_diversity"]:
            present = table[column].dropna()
            if not present.between(0, 1).all():
                raise ValueError(f"Invalid {method}/{column}")
        if not np.allclose(table.coverage + table.outlier_rate, 1):
            raise ValueError(f"Coverage mismatch for {method}")
        if table.loc[table.coherence.notna(), "coherence_definition"].nunique() > 1:
            raise ValueError(f"Mixed coherence definitions within {method}")
        pair_table = pairwise_assignment_ari(assignments)
        pair_table.insert(0, "method", method)
        saved_pairs = directory / "pairwise.csv"
        if saved_pairs.is_file():
            supplement = read(saved_pairs)
            merged = pair_table.merge(supplement, on=["method", "seed_a", "seed_b"],
                                      validate="one_to_one", suffixes=("", "_saved"))
            if len(merged) != len(pair_table) or not np.allclose(merged.ari_between_runs, merged.ari_between_runs_saved):
                raise ValueError(f"Saved pairwise ARI mismatch for {method}")
            for metric in ["keyword_jaccard", "document_cosine"]:
                if metric in merged:
                    pair_table[metric] = merged[metric]
        table["source_directory"] = str(directory).replace(chr(92), "/")
        runs.append(table)
        pairs.append(pair_table)
        status.append({"method": method, "status": "available", "n_seeds": len(table),
                       "seeds": ";".join(str(s) for s in sorted(table.seed)), "missing": ""})
    if not runs:
        raise ValueError("No complete stability submissions available")
    per_seed = pd.concat(runs, ignore_index=True)
    pairwise = pd.concat(pairs, ignore_index=True)
    run_summary = summarize_metrics(per_seed, ["topic_count", "coherence", "topic_diversity", "outlier_rate", "coverage"], "seed")
    pair_summary = summarize_metrics(pairwise, ["ari_between_runs", "ari_common_non_outliers", "common_non_outlier_fraction", "keyword_jaccard", "document_cosine"], "seed_pair")
    output.mkdir(parents=True, exist_ok=True)
    for name, data in {"per_seed.csv": per_seed, "pairwise.csv": pairwise,
                       "summary.csv": pd.concat([run_summary, pair_summary], ignore_index=True),
                       "coverage_status.csv": pd.DataFrame(status)}.items():
        data.to_csv(output / name, index=False, encoding="utf-8-sig")
    manifest = {"fit_split": "train", "evaluation_split": split, "n_documents": len(expected_ids),
                "outlier_policy": "all-document ARI includes -1; also report common-non-outlier ARI and retained fraction",
                "standard_deviation": "sample, ddof=1; descriptive only; seed pairs are dependent",
                "seed_policy": "retain submitted seed sets; no claim of shared-seed protocol",
                "sources_sha256": sources}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pd.DataFrame(status).to_string(index=False))
    print(pd.concat([run_summary, pair_summary], ignore_index=True).to_string(index=False))


if __name__ == "__main__":
    main()
