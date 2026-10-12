"""M11: phân cụm HDBSCAN, kiểm tra metadata và lưu kết quả có config_hash."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.dataset_loader import load_split_dataset
from src.features.embeddings import (
    load_embedding_artifact,
    build_embedding_metadata,
)
from src.models.hdbscan import reduce_embeddings, cluster_articles


CONFIG_PATH = Path("configs/hdbscan.yaml")
EMBEDDING_CONFIG_PATH = Path("configs/embeddings.yaml")


def get_config_hash(config: dict, embedding_config: dict) -> str:
    """Nhận diện điều kiện thí nghiệm chung của bảng summary."""
    payload = {
        "dataset": config["dataset"],
        "umap": config["umap"],
        "hdbscan": config["hdbscan"],
        "embedding_configs": {
            name: build_embedding_metadata(embedding_config, name)
            for name in config["embedding_artifacts"]
        },
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def read_compatible_summary(
    path: Path,
    config_hash: str,
) -> pd.DataFrame | None:
    """Đọc summary cũ và kiểm tra trước khi gộp kết quả."""
    if not path.exists():
        return None

    summary = pd.read_csv(path, encoding="utf-8-sig")
    required = {"embedding_run", "experiment", "config_hash"}

    if not required.issubset(summary.columns):
        raise ValueError(
            "Summary cũ chưa có thông tin cấu hình. "
            "Hãy chọn output_dir mới cho đợt chạy này."
        )

    if not summary["config_hash"].eq(config_hash).all():
        raise ValueError(
            "Summary thuộc cấu hình khác. "
            "Không được gộp với lần chạy hiện tại. "
            "Hãy chọn output_dir mới."
        )

    if summary.duplicated(["embedding_run", "experiment"]).any():
        raise ValueError("Summary cũ có cấu hình bị trùng")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chạy HDBSCAN trên một artifact embedding M10."
    )
    parser.add_argument(
        "--embedding",
        default="minilm_title_content",
        help="Tên embedding trong configs/hdbscan.yaml",
    )
    args = parser.parse_args()

    # 1. Đọc hai file cấu hình.
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    with EMBEDDING_CONFIG_PATH.open("r", encoding="utf-8") as file:
        embedding_config = yaml.safe_load(file)

    embedding_name = args.embedding
    artifact_paths = config["embedding_artifacts"]

    if embedding_name not in artifact_paths:
        raise ValueError(
            f"Embedding không có trong YAML M11: {embedding_name}. "
            f"Các lựa chọn: {list(artifact_paths)}"
        )

    if embedding_name not in embedding_config["runs"]:
        raise ValueError(
            f"Embedding không có trong YAML M10: {embedding_name}"
        )

    config_hash = get_config_hash(config, embedding_config)
    output_dir = Path(config["output_dir"])
    summary_path = output_dir / "m11_summary.csv"

    # Kiểm tra trước khi chạy UMAP để tránh tính toán không cần thiết.
    read_compatible_summary(summary_path, config_hash)

    # 2. Đọc tập train và artifact embedding.
    dataset_config = config["dataset"]
    train = load_split_dataset(
        dataset_config["path"],
        dataset_config["split_manifest_path"],
        dataset_config["split"],
    )
    embeddings = load_embedding_artifact(
        artifact_paths[embedding_name]
    )

    # 3. Kiểm tra đúng cấu hình tạo embedding.
    expected_metadata = build_embedding_metadata(
        embedding_config, embedding_name
    )
    actual_metadata = embeddings.attrs.get("embedding_metadata", {})

    if actual_metadata != expected_metadata:
        raise ValueError(
            f"{embedding_name}: metadata artifact không khớp "
            "configs/embeddings.yaml. Kiểm tra model, kiểu đầu vào, "
            "content_words, cột văn bản và split."
        )

    run_config = embedding_config["runs"][embedding_name]
    expected_version = embedding_config["dataset"]["data_version"]

    if not embeddings["model_name"].eq(run_config["model_name"]).all():
        raise ValueError("Tên model trong artifact không đúng")
    if not embeddings["data_version"].eq(expected_version).all():
        raise ValueError("Phiên bản dữ liệu trong artifact không đúng")

    expected_count = dataset_config["expected_articles"]
    article_ids = embeddings["article_id"].tolist()

    if len(train) != expected_count or len(embeddings) != expected_count:
        raise ValueError(
            f"Số bài không khớp: train={len(train)}, "
            f"embedding={len(embeddings)}, mong đợi={expected_count}"
        )

    if article_ids != train["article_id"].tolist():
        raise ValueError("ID hoặc thứ tự embedding không khớp tập train")

    vectors = np.stack(embeddings["embedding"].to_numpy())
    expected_shape = (
        expected_count,
        run_config["expected_dimensions"],
    )
    if vectors.shape != expected_shape:
        raise ValueError(
            f"Kích thước vector {vectors.shape}, "
            f"mong đợi {expected_shape}"
        )

    experiments = config["hdbscan"]["experiments"]
    names = [experiment["name"] for experiment in experiments]

    if not names or len(names) != len(set(names)):
        raise ValueError("Danh sách experiment rỗng hoặc tên bị trùng")

    print(f"Embedding: {embedding_name}")
    print(f"Kích thước vector: {vectors.shape}")
    print("Metadata: khớp cấu hình M10")
    print(f"Config hash: {config_hash}")
    print("Đang giảm chiều bằng UMAP...")

    # 4. Giảm chiều một lần cho embedding đang chạy.
    reduced = reduce_embeddings(
        embeddings,
        umap_params=config["umap"],
    )

    assignment_tables = []
    summary_rows = []

    # 5. Dùng chung kết quả UMAP cho các cấu hình HDBSCAN.
    for experiment in experiments:
        experiment_name = experiment["name"]

        result = cluster_articles(
            embeddings,
            reduced_vectors=reduced,
            hdbscan_params={
                "min_cluster_size": experiment["min_cluster_size"],
                "min_samples": experiment["min_samples"],
                "metric": config["hdbscan"]["metric"],
                "cluster_selection_method": experiment[
                    "cluster_selection_method"
                ],
            },
        )

        result["embedding_run"] = embedding_name
        result["experiment"] = experiment_name
        result["config_hash"] = config_hash
        assignment_tables.append(result)

        labels = result["cluster_id"].to_numpy()
        sizes = pd.Series(labels[labels != -1]).value_counts()
        n_noise = int(np.count_nonzero(labels == -1))
        n_clusters = len(sizes)

        summary_rows.append({
            "embedding_run": embedding_name,
            "experiment": experiment_name,
            "config_hash": config_hash,
            "min_cluster_size": experiment["min_cluster_size"],
            "min_samples": experiment["min_samples"],
            "cluster_selection_method": experiment[
                "cluster_selection_method"
            ],
            "n_articles": len(article_ids),
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_percent": round(100 * n_noise / len(article_ids), 2),
            "smallest_cluster": int(sizes.min()) if n_clusters else 0,
            "largest_cluster": int(sizes.max()) if n_clusters else 0,
        })

        print(
            f"{experiment_name}: {n_clusters} cụm, "
            f"{n_noise} bài nhiễu"
        )

    # 6. Gộp assignments của embedding này.
    assignments = pd.concat(assignment_tables, ignore_index=True)

    if assignments.duplicated(["experiment", "article_id"]).any():
        raise ValueError("Có ID trùng trong cùng một experiment")

    # 7. Cập nhật summary, giữ các embedding cùng cấu hình.
    summary = pd.DataFrame(summary_rows)
    previous = read_compatible_summary(summary_path, config_hash)

    if previous is not None:
        previous = previous.loc[
            previous["embedding_run"] != embedding_name
        ]
        summary = pd.concat([previous, summary], ignore_index=True)

    if summary.duplicated(["embedding_run", "experiment"]).any():
        raise ValueError("Bảng summary có cấu hình bị trùng")

    summary = summary.sort_values(
        ["embedding_run", "experiment"]
    ).reset_index(drop=True)

    # 8. Lưu kết quả.
    output_dir.mkdir(parents=True, exist_ok=True)
    assignment_path = (
        output_dir / f"{embedding_name}_assignments.csv"
    )

    assignments.to_csv(
        assignment_path,
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Đã lưu: {assignment_path} ({len(assignments)} dòng)")
    print(f"Đã cập nhật: {summary_path} ({len(summary)} dòng)")


if __name__ == "__main__":
    main()