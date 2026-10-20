"""Phân tích thay đổi nhãn giữa các seed E04 từ assignments đã lưu."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = (
    ROOT
    / "results/hdbscan/stability/"
      "minilm_title_content_eom_15_5_seeds42_51"
)
SEEDS = tuple(range(42, 52))

INPUTS = {
    "train": ("train_assignments.csv", 6493),
    "val": ("validation_assignments.csv", 1391),
}


def analyze_split(split: str, filename: str, expected_articles: int) -> None:
    data = pd.read_csv(
        RESULT_DIR / filename,
        encoding="utf-8-sig",
        dtype={"article_id": str},
    )

    required = {"article_id", "split", "umap_seed", "cluster_id"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{filename}: thiếu cột {sorted(missing)}")

    if not data["split"].eq(split).all():
        raise ValueError(f"{filename}: cột split không đúng")

    if set(data["umap_seed"]) != set(SEEDS):
        raise ValueError(f"{filename}: bộ seed không khớp {SEEDS}")

    if data.duplicated(["umap_seed", "article_id"]).any():
        raise ValueError(f"{filename}: trùng seed + article_id")

    by_seed = {}
    for seed in SEEDS:
        rows = data.loc[data["umap_seed"].eq(seed)].copy()
        if len(rows) != expected_articles:
            raise ValueError(
                f"{filename}, seed {seed}: cần {expected_articles} bài, "
                f"nhận {len(rows)}"
            )
        by_seed[seed] = rows.set_index("article_id")

    reference_ids = set(by_seed[SEEDS[0]].index)
    for seed in SEEDS[1:]:
        if set(by_seed[seed].index) != reference_ids:
            raise ValueError(
                f"{filename}: article_id không giống nhau giữa các seed"
            )

    # Cố định thứ tự ID để hai mảng nhãn luôn nói về cùng bài báo.
    article_ids = sorted(reference_ids)
    summary_rows = []
    transition_tables = []

    for seed_a, seed_b in combinations(SEEDS, 2):
        a = by_seed[seed_a].loc[article_ids, "cluster_id"].to_numpy(dtype=int)
        b = by_seed[seed_b].loc[article_ids, "cluster_id"].to_numpy(dtype=int)

        noise_a = a == -1
        noise_b = b == -1
        both_noise = noise_a & noise_b
        both_clustered = ~noise_a & ~noise_b
        noise_to_cluster = noise_a & ~noise_b
        cluster_to_noise = ~noise_a & noise_b
        noise_union = noise_a | noise_b

        n_both_clustered = int(both_clustered.sum())
        clustered_ari = (
            adjusted_rand_score(a[both_clustered], b[both_clustered])
            if n_both_clustered >= 2
            else np.nan
        )
        noise_jaccard = (
            float(both_noise.sum() / noise_union.sum())
            if noise_union.any()
            else 1.0
        )

        summary_rows.append({
            "split": split,
            "seed_a": seed_a,
            "seed_b": seed_b,
            "n_articles": len(article_ids),
            "both_noise": int(both_noise.sum()),
            "both_clustered": n_both_clustered,
            "noise_to_cluster": int(noise_to_cluster.sum()),
            "cluster_to_noise": int(cluster_to_noise.sum()),
            "n_noise_status_changed": int(
                noise_to_cluster.sum() + cluster_to_noise.sum()
            ),
            "noise_status_changed_percent": round(
                100 * (noise_to_cluster.sum() + cluster_to_noise.sum())
                / len(article_ids),
                2,
            ),
            "noise_jaccard": noise_jaccard,
            "ari_all_articles": adjusted_rand_score(a, b),
            "ari_both_clustered": clustered_ari,
        })

        changed = noise_to_cluster | cluster_to_noise
        changed_indices = np.flatnonzero(changed)
        transitions = pd.DataFrame({
            "split": split,
            "seed_a": seed_a,
            "seed_b": seed_b,
            "article_id": [
                article_ids[i] for i in changed_indices
            ],
            "cluster_a": a[changed_indices],
            "cluster_b": b[changed_indices],
            "direction": np.where(
                noise_to_cluster[changed_indices],
                "noise_to_cluster",
                "cluster_to_noise",
            ),
        })
        transition_tables.append(transitions)

    summary = pd.DataFrame(summary_rows)
    transitions = pd.concat(transition_tables, ignore_index=True)

    summary_path = RESULT_DIR / f"{split}_pairwise_state_analysis.csv"
    transitions_path = RESULT_DIR / f"{split}_transition_cases.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    transitions.to_csv(
        transitions_path, index=False, encoding="utf-8-sig"
    )

    print(f"\n{split.upper()}:")
    print(summary.to_string(index=False))
    print(f"Đã lưu: {summary_path}")
    print(f"Đã lưu: {transitions_path}")


def main() -> None:
    for split, (filename, expected_articles) in INPUTS.items():
        analyze_split(split, filename, expected_articles)


if __name__ == "__main__":
    main()