"""Đánh giá HDBSCAN M11 trên validation mà không fit trên validation."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer

from experiments.hdbscan.run_m11_hdbscan import get_config_hash
from src.data.dataset_loader import load_split_dataset
from src.features.embeddings import (
    build_embedding_metadata,
    create_embeddings,
    load_embedding_artifact,
)
from src.models.hdbscan import (
    fit_hdbscan_for_validation,
    fit_umap_for_validation,
    predict_hdbscan_validation,
)


EMBEDDING_RUN = "mpnet_title_content"
EXPERIMENT_NAMES = ("eom_15_5", "leaf_15_5")

M11_CONFIG_PATH = Path("configs/hdbscan.yaml")
M10_CONFIG_PATH = Path("configs/embeddings.yaml")

OUTPUT_DIR = Path("results/hdbscan/validation")
MODEL_DIR = Path("artifacts/hdbscan/validation")


def read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def verify_train_reproduction(
    train_ids: list[str],
    fitted_labels: np.ndarray,
    saved_assignments: pd.DataFrame,
    experiment_name: str,
    expected_hash: str,
) -> None:
    """Chỉ chấp nhận model tạo lại đúng nhãn train đã công bố ở M11."""
    old = saved_assignments.loc[
        saved_assignments["experiment"].eq(experiment_name)
    ].copy()

    if len(old) != len(train_ids):
        raise ValueError(f"{experiment_name}: số bài train không khớp M11")
    if old["article_id"].duplicated().any():
        raise ValueError(f"{experiment_name}: article_id train bị trùng")
    if set(old["article_id"]) != set(train_ids):
        raise ValueError(f"{experiment_name}: tập article_id train không khớp")
    if not old["config_hash"].eq(expected_hash).all():
        raise ValueError(f"{experiment_name}: config_hash không khớp M11")

    old_labels = (
        old.set_index("article_id")
        .loc[train_ids, "cluster_id"]
        .to_numpy(dtype=int)
    )

    if not np.array_equal(fitted_labels, old_labels):
        n_different = int(np.count_nonzero(fitted_labels != old_labels))
        raise ValueError(
            f"{experiment_name}: {n_different} nhãn train khác kết quả M11. "
            "Dừng để kiểm tra artifact, thứ tự ID và phiên bản thư viện."
        )


def main(embedding_run: str = EMBEDDING_RUN) -> None:
    m11 = read_yaml(M11_CONFIG_PATH)
    m10 = read_yaml(M10_CONFIG_PATH)
    config_hash = get_config_hash(m11, m10)

    dataset_config = m11["dataset"]
    if (
        embedding_run not in m10["runs"]
        or embedding_run not in m11["embedding_artifacts"]
    ):
        raise ValueError(f"Embedding run không có trong M10/M11: {embedding_run}")
    embedding_config = m10["runs"][embedding_run]

    train = load_split_dataset(
        dataset_config["path"],
        dataset_config["split_manifest_path"],
        "train",
    )
    train_embeddings = load_embedding_artifact(
        m11["embedding_artifacts"][embedding_run]
    )

    if len(train) != dataset_config["expected_articles"]:
        raise ValueError("Số bài train không khớp cấu hình M11")
    if train_embeddings["article_id"].tolist() != train["article_id"].tolist():
        raise ValueError("Thứ tự article_id của embedding train không khớp")
    if (
        train_embeddings.attrs.get("embedding_metadata")
        != build_embedding_metadata(m10, embedding_run)
    ):
        raise ValueError("Metadata embedding train không khớp cấu hình M10")

    train_vectors = np.stack(train_embeddings["embedding"].to_numpy())
    if train_vectors.shape != (
        len(train),
        embedding_config["expected_dimensions"],
    ):
        raise ValueError("Kích thước embedding train không đúng")

    assignments_path = (
        Path(m11["output_dir"]) / f"{embedding_run}_assignments.csv"
    )
    old_assignments = pd.read_csv(
        assignments_path,
        encoding="utf-8-sig",
        dtype={"article_id": str},
    )

    experiments_by_name = {
        item["name"]: item
        for item in m11["hdbscan"]["experiments"]
    }

    print(f"Fit UMAP trên train: {embedding_run}...")
    reducer, train_5d = fit_umap_for_validation(
        train_embeddings,
        m11["umap"],
    )

    # Kiểm tra model train trước khi mã hóa validation hoặc ghi file.
    fitted_clusterers = {}
    for name in EXPERIMENT_NAMES:
        if name not in experiments_by_name:
            raise ValueError(f"Không tìm thấy experiment: {name}")

        experiment = experiments_by_name[name]
        clusterer, train_labels = fit_hdbscan_for_validation(
            train_5d,
            {
                "min_cluster_size": experiment["min_cluster_size"],
                "min_samples": experiment["min_samples"],
                "metric": m11["hdbscan"]["metric"],
                "cluster_selection_method": experiment[
                    "cluster_selection_method"
                ],
            },
        )

        verify_train_reproduction(
            train["article_id"].tolist(),
            train_labels,
            old_assignments,
            name,
            config_hash,
        )
        fitted_clusterers[name] = clusterer
        print(f"{name}: nhãn train khớp kết quả M11")

    val = load_split_dataset(
        dataset_config["path"],
        dataset_config["split_manifest_path"],
        "val",
    ).reset_index(drop=True)

    if len(val) != 1391:
        raise ValueError(f"Validation phải có 1391 bài, nhận {len(val)}")
    if val["article_id"].duplicated().any():
        raise ValueError("Validation có article_id trùng")
    if set(val["article_id"]) & set(train["article_id"]):
        raise ValueError("Train và validation có article_id giao nhau")

    print(f"Đang mã hóa {len(val)} bài validation...")
    encoder = SentenceTransformer(embedding_config["model_name"])

    val_embeddings = create_embeddings(
        val,
        encoder,
        model_name=embedding_config["model_name"],
        data_version=m10["dataset"]["data_version"],
        content_words=embedding_config["content_words"],
        title_col=m10["dataset"]["title_col"],
        content_col=m10["dataset"]["content_col"],
        batch_size=m10["encoding"]["batch_size"],
    )

    val_vectors = np.stack(val_embeddings["embedding"].to_numpy())
    if val_vectors.shape != (
        len(val),
        embedding_config["expected_dimensions"],
    ):
        raise ValueError("Kích thước embedding validation không đúng")

    predictions = {}
    for name, clusterer in fitted_clusterers.items():
        result = predict_hdbscan_validation(
            val_embeddings,
            reducer,
            clusterer,
        )
        if result["article_id"].tolist() != val["article_id"].tolist():
            raise ValueError(f"{name}: thứ tự article_id validation bị đổi")

        result.insert(1, "split", "val")
        result["embedding_run"] = embedding_run
        result["experiment"] = name
        result["config_hash"] = config_hash
        predictions[name] = result

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    reducer_path = MODEL_DIR / f"{embedding_run}_umap.joblib"
    joblib.dump(reducer, reducer_path)
    print(f"Đã lưu: {reducer_path}")

    for name, result in predictions.items():
        clusterer_path = MODEL_DIR / (
            f"{embedding_run}_{name}_hdbscan.joblib"
        )
        result_path = OUTPUT_DIR / (
            f"{embedding_run}_{name}_val_assignments.csv"
        )

        joblib.dump(fitted_clusterers[name], clusterer_path)
        result.to_csv(
            result_path,
            index=False,
            encoding="utf-8-sig",
        )

        n_noise = int(result["cluster_id"].eq(-1).sum())
        noise_percent = 100 * n_noise / len(result)
        print(
            f"{name}: {len(result)} bài val, "
            f"{n_noise} bài -1 ({noise_percent:.2f}%)"
        )
        print(f"Đã lưu: {clusterer_path}")
        print(f"Đã lưu: {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-run", default=EMBEDDING_RUN)
    args = parser.parse_args()
    main(args.embedding_run)