"""E04: kiểm tra độ ổn định dự đoán HDBSCAN trên validation."""

from itertools import combinations

import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.metrics import adjusted_rand_score

from experiments.hdbscan.run_e04_train_stability import (
    EMBEDDING_RUN,
    EXPERIMENT,
    OUTPUT_DIR,
    ROOT,
    SEEDS,
    load_inputs,
    load_original_labels,
)
from src.data.dataset_loader import load_split_dataset
from src.features.embeddings import create_embeddings
from src.models.hdbscan import (
    fit_hdbscan_for_validation,
    fit_umap_for_validation,
    predict_hdbscan_validation,
)


def load_validation_embeddings(m11_config):
    """Mã hóa validation một lần, dùng chung cho mọi seed."""
    with (ROOT / "configs/embeddings.yaml").open(encoding="utf-8") as file:
        m10_config = yaml.safe_load(file)

    run_config = m10_config["runs"][EMBEDDING_RUN]
    dataset_config = m11_config["dataset"]

    validation = load_split_dataset(
        ROOT / dataset_config["path"],
        ROOT / dataset_config["split_manifest_path"],
        "val",
    ).reset_index(drop=True)

    if len(validation) != 1391:
        raise ValueError(f"Cần 1391 bài validation, nhận {len(validation)}")
    if validation["article_id"].duplicated().any():
        raise ValueError("Validation có article_id trùng")

    print(f"Đang mã hóa {len(validation)} bài validation...")
    encoder = SentenceTransformer(run_config["model_name"])
    embeddings = create_embeddings(
        validation,
        encoder,
        model_name=run_config["model_name"],
        data_version=m10_config["dataset"]["data_version"],
        content_words=run_config["content_words"],
        title_col=m10_config["dataset"]["title_col"],
        content_col=m10_config["dataset"]["content_col"],
        batch_size=m10_config["encoding"]["batch_size"],
    )

    ids = validation["article_id"].astype(str).tolist()
    if embeddings["article_id"].astype(str).tolist() != ids:
        raise ValueError("Thứ tự article_id validation bị thay đổi")

    return ids, embeddings


def load_original_validation_labels(article_ids):
    """Nhãn validation M11 đã lưu để đối chiếu seed 42."""
    path = (
        ROOT
        / "results/hdbscan/validation"
        / f"{EMBEDDING_RUN}_{EXPERIMENT}_val_assignments.csv"
    )
    original = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"article_id": str},
    )

    if len(original) != len(article_ids):
        raise ValueError("Số nhãn validation M11 không khớp")
    if original["article_id"].duplicated().any():
        raise ValueError("Nhãn validation M11 có article_id trùng")
    if set(original["article_id"]) != set(article_ids):
        raise ValueError("Tập article_id validation không khớp M11")

    return (
        original.set_index("article_id")
        .loc[article_ids, "cluster_id"]
        .to_numpy(dtype=int)
    )


def main():
    m11_config, train_embeddings, train_ids, experiment = load_inputs()
    original_train_labels = load_original_labels(train_ids)

    val_ids, val_embeddings = load_validation_embeddings(m11_config)
    if set(train_ids) & set(val_ids):
        raise ValueError("Train và validation có article_id giao nhau")

    original_val_labels = load_original_validation_labels(val_ids)

    hdbscan_params = {
        "min_cluster_size": experiment["min_cluster_size"],
        "min_samples": experiment["min_samples"],
        "metric": m11_config["hdbscan"]["metric"],
        "cluster_selection_method": experiment[
            "cluster_selection_method"
        ],
    }

    metrics_rows = []
    pairwise_rows = []
    assignment_tables = []
    labels_by_seed = {}

    for seed in SEEDS:
        print(f"Đang fit UMAP trên train với seed {seed}...")
        umap_params = {
            **m11_config["umap"],
            "random_state": seed,
        }
        reducer, train_5d = fit_umap_for_validation(
            train_embeddings,
            umap_params,
        )
        clusterer, train_labels = fit_hdbscan_for_validation(
            train_5d,
            hdbscan_params,
        )

        if seed == 42 and not np.array_equal(
            train_labels, original_train_labels
        ):
            raise ValueError(
                "Seed 42 không tạo lại đúng nhãn train M11"
            )

        # Chỉ transform và predict; không fit trên validation.
        predicted = predict_hdbscan_validation(
            val_embeddings,
            reducer,
            clusterer,
        )
        if predicted["article_id"].astype(str).tolist() != val_ids:
            raise ValueError(
                f"Seed {seed}: thứ tự article_id validation bị đổi"
            )

        labels = predicted["cluster_id"].to_numpy(dtype=int)
        if seed == 42 and not np.array_equal(
            labels, original_val_labels
        ):
            raise ValueError(
                "Seed 42 không tạo lại đúng nhãn validation M11"
            )

        labels_by_seed[seed] = labels
        n_noise = int(np.count_nonzero(labels == -1))
        noise_percent = 100 * n_noise / len(val_ids)

        metrics_rows.append({
            "embedding_run": EMBEDDING_RUN,
            "experiment": EXPERIMENT,
            "split": "val",
            "umap_seed": seed,
            "n_articles": len(val_ids),
            "n_noise": n_noise,
            "noise_percent": round(noise_percent, 2),
            "coverage_percent": round(100 - noise_percent, 2),
        })

        predicted.insert(1, "split", "val")
        predicted["embedding_run"] = EMBEDDING_RUN
        predicted["experiment"] = EXPERIMENT
        predicted["umap_seed"] = seed
        assignment_tables.append(predicted)

        print(
            f"Seed {seed}: {n_noise} bài -1 "
            f"({noise_percent:.2f}%)"
        )

    for seed_a, seed_b in combinations(SEEDS, 2):
        pairwise_rows.append({
            "embedding_run": EMBEDDING_RUN,
            "experiment": EXPERIMENT,
            "split": "val",
            "seed_a": seed_a,
            "seed_b": seed_b,
            "n_articles": len(val_ids),
            "pairwise_ari": adjusted_rand_score(
                labels_by_seed[seed_a],
                labels_by_seed[seed_b],
            ),
        })

    metrics = pd.DataFrame(metrics_rows)
    pairwise = pd.DataFrame(pairwise_rows)
    assignments = pd.concat(assignment_tables, ignore_index=True)

    summary_rows = []
    for column in ("n_noise", "noise_percent", "coverage_percent"):
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
    for filename, table in (
        ("validation_seed_metrics.csv", metrics),
        ("validation_pairwise_ari.csv", pairwise),
        ("validation_assignments.csv", assignments),
        ("validation_summary.csv", summary),
    ):
        table.to_csv(
            OUTPUT_DIR / filename,
            index=False,
            encoding="utf-8-sig",
        )

    print("\nKết quả từng seed trên validation:")
    print(metrics.to_string(index=False))
    print("\nARI giữa các cặp seed trên validation:")
    print(pairwise.to_string(index=False))
    print("\nTóm tắt:")
    print(summary.to_string(index=False))
    print(f"\nĐã lưu CSV trong: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()