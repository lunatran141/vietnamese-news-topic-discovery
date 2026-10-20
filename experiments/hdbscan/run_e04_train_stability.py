"""E04: kiểm tra độ ổn định HDBSCAN khi thay seed của UMAP."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score

from src.data.dataset_loader import load_split_dataset
from src.features.embeddings import (
    build_embedding_metadata,
    load_embedding_artifact,
)
from src.models.hdbscan import cluster_articles, reduce_embeddings


ROOT = Path(__file__).resolve().parents[2]

EMBEDDING_RUN = "minilm_title_content"
EXPERIMENT = "eom_15_5"
SEEDS = tuple(range(42, 52))
OUTPUT_DIR = (
    ROOT
    / "results/hdbscan/stability"
    / f"{EMBEDDING_RUN}_{EXPERIMENT}_seeds{SEEDS[0]}_{SEEDS[-1]}"
)

def load_inputs():
    with (ROOT / "configs/hdbscan.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    with (ROOT / "configs/embeddings.yaml").open(
        encoding="utf-8"
    ) as file:
        embedding_config = yaml.safe_load(file)

    dataset_config = config["dataset"]
    train = load_split_dataset(
        ROOT / dataset_config["path"],
        ROOT / dataset_config["split_manifest_path"],
        "train",
    )

    embedding_path = (
        ROOT / config["embedding_artifacts"][EMBEDDING_RUN]
    )
    embeddings = load_embedding_artifact(embedding_path)

    expected_count = int(dataset_config["expected_articles"])
    if len(train) != expected_count:
        raise ValueError("Số bài train không khớp config M11")
    if len(embeddings) != expected_count:
        raise ValueError("Số embedding không khớp config M11")

    article_ids = train["article_id"].astype(str).tolist()
    embedding_ids = embeddings["article_id"].astype(str).tolist()
    if embedding_ids != article_ids:
        raise ValueError(
            "ID hoặc thứ tự embedding không khớp tập train"
        )

    expected_metadata = build_embedding_metadata(
        embedding_config,
        EMBEDDING_RUN,
    )
    actual_metadata = embeddings.attrs.get(
        "embedding_metadata"
    )
    if actual_metadata != expected_metadata:
        raise ValueError(
            "Metadata embedding không khớp configs/embeddings.yaml"
        )

    experiment = next(
        (
            item
            for item in config["hdbscan"]["experiments"]
            if item["name"] == EXPERIMENT
        ),
        None,
    )
    if experiment is None:
        raise ValueError(
            f"Không tìm thấy experiment {EXPERIMENT}"
        )

    return config, embeddings, article_ids, experiment


def load_original_labels(article_ids):
    """Lấy nhãn M11 gốc để kiểm tra lại seed 42."""
    path = (
        ROOT
        / f"results/hdbscan/{EMBEDDING_RUN}_assignments.csv"
    )
    original = pd.read_csv(
        path,
        dtype={"article_id": str},
    )
    original = original.loc[
        original["experiment"].eq(EXPERIMENT)
    ].copy()

    if len(original) != len(article_ids):
        raise ValueError(
            "Số assignments M11 gốc không khớp train"
        )
    if original["article_id"].duplicated().any():
        raise ValueError(
            "Assignments M11 gốc có article_id trùng"
        )
    if set(original["article_id"]) != set(article_ids):
        raise ValueError(
            "Assignments M11 gốc không khớp ID train"
        )

    original = original.set_index("article_id")
    return original.loc[
        article_ids, "cluster_id"
    ].to_numpy(dtype=int)


def main():
    config, embeddings, article_ids, experiment = load_inputs()
    original_labels = load_original_labels(article_ids)

    hdbscan_params = {
        "min_cluster_size": experiment["min_cluster_size"],
        "min_samples": experiment["min_samples"],
        "metric": config["hdbscan"]["metric"],
        "cluster_selection_method": (
            experiment["cluster_selection_method"]
        ),
    }

    metrics_rows = []
    assignment_tables = []
    labels_by_seed = {}

    for seed in SEEDS:
        print(f"Đang chạy UMAP với seed {seed}...")

        umap_params = {
            **config["umap"],
            "random_state": seed,
        }
        reduced_vectors = reduce_embeddings(
            embeddings,
            umap_params=umap_params,
        )

        assigned = cluster_articles(
            embeddings,
            reduced_vectors=reduced_vectors,
            hdbscan_params=hdbscan_params,
        )
        if assigned["article_id"].astype(str).tolist() != article_ids:
            raise ValueError(
                f"Seed {seed}: thứ tự article_id bị thay đổi"
            )

        labels = assigned["cluster_id"].to_numpy(dtype=int)

        # Seed gốc phải tạo lại đúng kết quả M11 đã lưu.
        if seed == 42 and not np.array_equal(
            labels,
            original_labels,
        ):
            raise ValueError(
                "Seed 42 không khớp nhãn M11 gốc. "
                "Kiểm tra cấu hình và phiên bản môi trường."
            )

        labels_by_seed[seed] = labels

        n_noise = int(np.count_nonzero(labels == -1))
        n_clusters = len(set(labels) - {-1})
        noise_percent = 100 * n_noise / len(article_ids)

        metrics_rows.append({
            "embedding_run": EMBEDDING_RUN,
            "experiment": EXPERIMENT,
            "split": "train",
            "umap_seed": seed,
            "n_articles": len(article_ids),
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_percent": round(noise_percent, 2),
        })

        assigned.insert(1, "split", "train")
        assigned["embedding_run"] = EMBEDDING_RUN
        assigned["experiment"] = EXPERIMENT
        assigned["umap_seed"] = seed
        assignment_tables.append(assigned)

        print(
            f"Seed {seed}: {n_clusters} cụm, "
            f"{n_noise} bài -1 ({noise_percent:.2f}%)"
        )

    pairwise_rows = []
    for seed_a, seed_b in combinations(SEEDS, 2):
        pairwise_rows.append({
            "embedding_run": EMBEDDING_RUN,
            "experiment": EXPERIMENT,
            "split": "train",
            "seed_a": seed_a,
            "seed_b": seed_b,
            "n_articles": len(article_ids),
            "pairwise_ari": adjusted_rand_score(
                labels_by_seed[seed_a],
                labels_by_seed[seed_b],
            ),
        })

    metrics = pd.DataFrame(metrics_rows)
    pairwise = pd.DataFrame(pairwise_rows)
    assignments = pd.concat(
        assignment_tables,
        ignore_index=True,
    )

    summary_rows = []
    for column in ("n_clusters", "n_noise", "noise_percent"):
        values = metrics[column]
        summary_rows.append({
            "metric": column,
            "mean": values.mean(),
            "std": values.std(ddof=1),
            "n_values": len(values),
        })

    ari_values = pairwise["pairwise_ari"]
    summary_rows.append({
        "metric": "pairwise_ari",
        "mean": ari_values.mean(),
        "std": ari_values.std(ddof=1),
        "n_values": len(ari_values),
    })
    summary = pd.DataFrame(summary_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(
        OUTPUT_DIR / "train_seed_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pairwise.to_csv(
        OUTPUT_DIR / "train_pairwise_ari.csv",
        index=False,
        encoding="utf-8-sig",
    )
    assignments.to_csv(
        OUTPUT_DIR / "train_assignments.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        OUTPUT_DIR / "train_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nKết quả từng seed:")
    print(metrics.to_string(index=False))
    print("\nARI giữa các cặp seed:")
    print(pairwise.to_string(index=False))
    print("\nTóm tắt:")
    print(summary.to_string(index=False))
    print(f"\nĐã lưu CSV trong: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()