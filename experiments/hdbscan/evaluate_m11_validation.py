"""Tính metric validation từ assignments M11 đã lưu, không chạy lại mô hình."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


MANIFEST_PATH = Path("data/splits/split_manifest.csv")
ASSIGNMENTS_DIR = Path("results/hdbscan/validation")
OUTPUT_PATH = ASSIGNMENTS_DIR / "m11_validation_metrics.csv"

EMBEDDING_RUNS = ("mpnet_title_content", "minilm_title_content")
EXPERIMENTS = ("eom_15_5", "leaf_15_5")
EXPECTED_VAL_COUNT = 1391


def evaluate_one_run(
    val: pd.DataFrame,
    embedding_run: str,
    experiment: str,
) -> dict:
    path = ASSIGNMENTS_DIR / (
        f"{embedding_run}_{experiment}_val_assignments.csv"
    )
    assignments = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"article_id": str},
    )

    run_name = f"{embedding_run}/{experiment}"

    if len(assignments) != EXPECTED_VAL_COUNT:
        raise ValueError(f"{run_name}: sai số dòng assignments")
    if (
        assignments["article_id"].isna().any()
        or assignments["article_id"].duplicated().any()
    ):
        raise ValueError(f"{run_name}: article_id bị thiếu hoặc trùng")
    if set(assignments["article_id"]) != set(val["article_id"]):
        raise ValueError(f"{run_name}: article_id không khớp validation")
    if not assignments["split"].eq("val").all():
        raise ValueError(f"{run_name}: cột split không phải val")
    if not assignments["embedding_run"].eq(embedding_run).all():
        raise ValueError(f"{run_name}: embedding_run không khớp")
    if not assignments["experiment"].eq(experiment).all():
        raise ValueError(f"{run_name}: experiment không khớp")
    if assignments["cluster_id"].isna().any():
        raise ValueError(f"{run_name}: thiếu cluster_id")
    if assignments["config_hash"].isna().any():
        raise ValueError(f"{run_name}: thiếu config_hash")

    hashes = assignments["config_hash"].unique()
    if len(hashes) != 1:
        raise ValueError(f"{run_name}: có nhiều config_hash")

    # Giữ thứ tự assignments; ghép category_gold theo article_id.
    paired = assignments.merge(
        val,
        on="article_id",
        how="left",
        validate="one_to_one",
    )

    y_true = paired["category_gold"]
    y_pred = paired["cluster_id"]
    n_articles = len(paired)
    n_noise = int(y_pred.eq(-1).sum())

    return {
        "embedding_run": embedding_run,
        "experiment": experiment,
        "split": "val",
        "n_articles": n_articles,
        "n_noise": n_noise,
        "noise_percent": round(100 * n_noise / n_articles, 2),
        "coverage_percent": round(
            100 * (n_articles - n_noise) / n_articles, 2
        ),
        "ari": round(adjusted_rand_score(y_true, y_pred), 4),
        "nmi": round(normalized_mutual_info_score(y_true, y_pred), 4),
        "config_hash": hashes[0],
    }


def main() -> None:
    manifest = pd.read_csv(
        MANIFEST_PATH,
        encoding="utf-8-sig",
        dtype={"article_id": str},
    )
    val = manifest.loc[
        manifest["split"].eq("val"),
        ["article_id", "category_gold"],
    ].copy()

    if len(val) != EXPECTED_VAL_COUNT:
        raise ValueError(
            f"Validation có {len(val)} bài, cần {EXPECTED_VAL_COUNT}"
        )
    if (
        val["article_id"].isna().any()
        or val["article_id"].duplicated().any()
    ):
        raise ValueError("article_id validation bị thiếu hoặc trùng")
    if val["category_gold"].isna().any():
        raise ValueError("Validation có category_gold bị thiếu")

    rows = []
    for embedding_run in EMBEDDING_RUNS:
        for experiment in EXPERIMENTS:
            rows.append(evaluate_one_run(val, embedding_run, experiment))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(metrics.to_string(index=False))
    print(f"Đã lưu: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()