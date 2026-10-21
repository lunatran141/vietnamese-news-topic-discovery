"""Đo runtime cho run HDBSCAN minilm_title_content_eom_15_5."""

import json
import os
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
import torch
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from experiments.hdbscan.run_m11_hdbscan import get_config_hash
from src.data.dataset_loader import load_split_dataset
from src.features.embeddings import create_embeddings
from src.models.hdbscan import (
    fit_hdbscan_for_validation,
    fit_umap_for_validation,
    predict_hdbscan_validation,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/hdbscan/benchmark_p01_runtime.json"
RUN_ID = "minilm_title_content_eom_15_5"


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def synchronize_gpu() -> None:
    """Chờ tác vụ GPU hoàn tất trước khi đọc đồng hồ."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    m10 = read_yaml(ROOT / "configs/embeddings.yaml")
    m11 = read_yaml(ROOT / "configs/hdbscan.yaml")

    embedding_run = m10["runs"]["minilm_title_content"]
    experiment = next(
        item
        for item in m11["hdbscan"]["experiments"]
        if item["name"] == "eom_15_5"
    )

    dataset = m11["dataset"]
    train = load_split_dataset(
        ROOT / dataset["path"],
        ROOT / dataset["split_manifest_path"],
        "train",
    ).reset_index(drop=True)
    val = load_split_dataset(
        ROOT / dataset["path"],
        ROOT / dataset["split_manifest_path"],
        "val",
    ).reset_index(drop=True)

    if len(train) != 6493 or len(val) != 1391:
        raise ValueError("Số bài train/validation không đúng")
    if set(train["article_id"]) & set(val["article_id"]):
        raise ValueError("Train và validation trùng article_id")
    if val["category_gold"].isna().any():
        raise ValueError("Validation thiếu category_gold")

    # Đo tải model riêng. Runtime chính bên dưới không gồm giai đoạn này.
    model_load_start = perf_counter()
    encoder = SentenceTransformer(embedding_run["model_name"])
    synchronize_gpu()
    model_load_seconds = perf_counter() - model_load_start

    def encode(articles):
        return create_embeddings(
            articles,
            encoder,
            model_name=embedding_run["model_name"],
            data_version=m10["dataset"]["data_version"],
            content_words=embedding_run["content_words"],
            title_col=m10["dataset"]["title_col"],
            content_col=m10["dataset"]["content_col"],
            batch_size=m10["encoding"]["batch_size"],
        )

    t0 = perf_counter()
    train_embeddings = encode(train)
    synchronize_gpu()
    t1 = perf_counter()

    reducer, train_5d = fit_umap_for_validation(
        train_embeddings,
        m11["umap"],
    )
    t2 = perf_counter()

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
    t3 = perf_counter()

    val_embeddings = encode(val)
    synchronize_gpu()
    t4 = perf_counter()

    # Hàm này gồm UMAP transform và HDBSCAN approximate_predict.
    val_predictions = predict_hdbscan_validation(
        val_embeddings,
        reducer,
        clusterer,
    )
    t5 = perf_counter()

    train_clusters = len(set(train_labels) - {-1})
    train_outliers = int(np.count_nonzero(train_labels == -1))
    val_outliers = int(val_predictions["cluster_id"].eq(-1).sum())

    if (train_clusters, train_outliers, val_outliers) != (
        101,
        1608,
        484,
    ):
        raise ValueError(
            "Benchmark không tái tạo đúng run P01: "
            f"{train_clusters} cụm, "
            f"{train_outliers} train outlier, "
            f"{val_outliers} validation outlier"
        )

    # Giữ nguyên tên khóa: exporter P01 và test hiện đang đọc các khóa này.
    phases = {
        "train_representation_seconds": t1 - t0,
        "umap_fit_seconds": t2 - t1,
        "hdbscan_fit_seconds": t3 - t2,
        "val_representation_seconds": t4 - t3,
        "val_predict_seconds": t5 - t4,
    }

    if not all(np.isfinite(value) and value > 0 for value in phases.values()):
        raise ValueError("Có thời gian giai đoạn không hợp lệ")

    # Đo đánh giá validation riêng, ngoài runtime chính.
    evaluation_start = perf_counter()

    paired = val[["article_id", "category_gold"]].merge(
        val_predictions[["article_id", "cluster_id"]],
        on="article_id",
        how="inner",
        validate="one_to_one",
    )
    if len(paired) != len(val):
        raise ValueError("Dự đoán validation không khớp article_id")

    validation_ari = adjusted_rand_score(
        paired["category_gold"],
        paired["cluster_id"],
    )
    validation_nmi = normalized_mutual_info_score(
        paired["category_gold"],
        paired["cluster_id"],
    )
    validation_coverage = 1 - val_outliers / len(val)

    evaluation_seconds = perf_counter() - evaluation_start

    result = {
        "run_id": RUN_ID,
        "config_hash": get_config_hash(m11, m10),
        "train_count": len(train),
        "val_count": len(val),
        "n_repeats": 1,
        "measurement_note": (
            "Một lần đo (n_repeats=1). Chỉ so sánh runtime với "
            "run có cùng phạm vi đo và điều kiện máy."
        ),
        "runtime_scope": (
            "MiniLM encode train và val + UMAP 5D fit train "
            "+ HDBSCAN fit train + UMAP transform và "
            "approximate_predict val; không gồm tải model, "
            "đánh giá metric, đọc/ghi file và xuất dữ liệu UI"
        ),
        # Chỉ cộng năm giai đoạn chính; không cộng lại các alias bên dưới.
        "runtime_seconds": sum(phases.values()),
        "phases": phases,
        "model_load_seconds": model_load_seconds,
        "representation_seconds": (
            phases["train_representation_seconds"]
            + phases["val_representation_seconds"]
        ),
        "validation_predict_seconds": phases["val_predict_seconds"],
        # Chưa dự đoán test nên để null trong JSON.
        "test_predict_seconds": None,
        "evaluation_seconds": evaluation_seconds,
        "evaluation_scope": "ARI, NMI và coverage trên validation",
        "validation_metrics": {
            "ari": validation_ari,
            "nmi": validation_nmi,
            "coverage": validation_coverage,
        },
        "environment": {
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "ram_gb": round(
                psutil.virtual_memory().total / (1024**3),
                2,
            ),
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "python_version": platform.python_version(),
            "library_versions": {
                name: version(name)
                for name in (
                    "numpy",
                    "umap-learn",
                    "hdbscan",
                    "sentence-transformers",
                    "torch",
                )
            },
            "batch_size": m10["encoding"]["batch_size"],
            "model_loaded_before_timing": True,
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Đã lưu: {OUTPUT}")
    print(f"Runtime chính: {result['runtime_seconds']:.2f} giây")
    print(f"Tải model: {model_load_seconds:.2f} giây")
    print(f"Đánh giá validation: {evaluation_seconds:.2f} giây")
    for name, seconds in phases.items():
        print(f"{name}: {seconds:.2f} giây")


if __name__ == "__main__":
    main()