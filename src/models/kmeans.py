"""Module triển khai mô hình phân cụm K-Means cho bài báo tiếng Việt (Task M02).

Cung cấp các hàm huấn luyện, dự đoán, tính toán khoảng cách đến tâm,
đo lường các chỉ số và lưu trữ artifacts.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.cluster import contingency_matrix


def train_kmeans(
    train_matrix: csr_matrix | np.ndarray,
    n_clusters: int,
    *,
    init: str = "k-means++",
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int = 42,
) -> KMeans:
    """Huấn luyện mô hình K-Means trên ma trận đặc trưng huấn luyện."""
    if train_matrix.shape[0] == 0:
        raise ValueError("Ma trận huấn luyện rỗng")

    if n_clusters < 2:
        raise ValueError("Số cụm n_clusters phải lớn hơn hoặc bằng 2")

    if n_clusters > train_matrix.shape[0]:
        raise ValueError(
            f"Số cụm n_clusters ({n_clusters}) không thể vượt quá số mẫu ({train_matrix.shape[0]})"
        )

    model = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(train_matrix)
    return model


def predict_kmeans(
    model: KMeans,
    matrix: csr_matrix | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Dự đoán cluster_id và tính khoảng cách Euclid chính xác đến tâm cụm."""
    if matrix.shape[0] == 0:
        raise ValueError("Ma trận dự đoán rỗng")

    # tính khoảng cách từ mọi bài báo đến tất cả các tâm cụm
    dist_matrix = model.transform(matrix)

    # cluster_id là cụm có khoảng cách nhỏ nhất
    cluster_ids = np.argmin(dist_matrix, axis=1)

    # Khoảng cách đến tâm cụm được gán
    distances_to_center = dist_matrix[np.arange(len(cluster_ids)), cluster_ids]

    return cluster_ids, distances_to_center


def compute_stability(label_sets: list[np.ndarray | list[int]]) -> tuple[float, float]:
    """Đo độ ổn định của phân cụm qua nhiều seed."""
    if len(label_sets) < 2:
        raise ValueError("Cần ít nhất 2 bộ nhãn để đo độ ổn định")

    scores = [
        adjusted_rand_score(first, second)
        for first, second in itertools.combinations(label_sets, 2)
    ]
    return float(np.mean(scores)), float(np.std(scores))


def compute_purity(y_true: np.ndarray | list, y_pred: np.ndarray | list) -> float:
    """Tính chỉ số Purity của phân cụm so với nhãn chuẩn."""
    cont_matrix = contingency_matrix(y_true, y_pred)
    total_samples = np.sum(cont_matrix)
    if total_samples == 0:
        return 0.0
    return float(np.sum(np.amax(cont_matrix, axis=0)) / total_samples)


def compute_clustering_metrics(
    matrix: csr_matrix | np.ndarray,
    cluster_ids: np.ndarray | list[int],
    *,
    model: KMeans | None = None,
    gold_labels: np.ndarray | list | None = None,
) -> dict[str, float]:
    """Đo lường toàn diện các chỉ số phân cụm nội tại và ngoại tại."""
    labels = np.asarray(cluster_ids)
    n_samples = matrix.shape[0]
    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters)

    metrics: dict[str, float] = {}

    # Inertia
    if model is not None and hasattr(model, "inertia_"):
        metrics["inertia"] = float(model.inertia_)
    else:
        metrics["inertia"] = 0.0

    # Các chỉ số nội tại yêu cầu tối thiểu 2 cụm và nhỏ hơn n_samples
    if 1 < n_clusters < n_samples:
        # Silhouette Score
        metrics["silhouette_score"] = float(
            silhouette_score(matrix, labels, metric="euclidean")
        )

        # Davies-Bouldin và Calinski-Harabasz yêu cầu dense array
        dense_matrix = matrix.toarray() if issparse(matrix) else matrix
        metrics["davies_bouldin_index"] = float(
            davies_bouldin_score(dense_matrix, labels)
        )
        metrics["calinski_harabasz_score"] = float(
            calinski_harabasz_score(dense_matrix, labels)
        )
    else:
        metrics["silhouette_score"] = 0.0
        metrics["davies_bouldin_index"] = 0.0
        metrics["calinski_harabasz_score"] = 0.0

    # Các chỉ số ngoại vi so khớp với category_gold
    if gold_labels is not None:
        valid_gold = np.asarray(gold_labels)
        if len(valid_gold) == len(labels):
            metrics["ari"] = float(adjusted_rand_score(valid_gold, labels))
            metrics["nmi"] = float(normalized_mutual_info_score(valid_gold, labels))
            metrics["purity"] = float(compute_purity(valid_gold, labels))

    return {k: round(v, 4) for k, v in metrics.items()}


def extract_cluster_keywords(
    model: KMeans,
    feature_names: list[str],
    top_n: int = 15,
) -> dict[str, list[dict[str, Any]]]:
    """Trích xuất từ khóa tiêu biểu của từng cụm dựa vào trọng số tâm cụm (centroid)."""
    centroids = model.cluster_centers_
    n_clusters = centroids.shape[0]
    n_features = len(feature_names)

    if centroids.shape[1] != n_features:
        raise ValueError(
            f"Kích thước tâm cụm ({centroids.shape[1]}) không khớp số feature names ({n_features})"
        )

    cluster_keywords: dict[str, list[dict[str, Any]]] = {}

    for k in range(n_clusters):
        centroid_vector = centroids[k]
        top_indices = np.argsort(centroid_vector)[::-1][:top_n]

        keywords = [
            {
                "term": feature_names[idx],
                "weight": round(float(centroid_vector[idx]), 4),
            }
            for idx in top_indices
        ]
        cluster_keywords[str(k)] = keywords

    return cluster_keywords


def find_representative_docs(
    cluster_ids: np.ndarray | list[int],
    distances_to_center: np.ndarray | list[float],
    article_ids: list[str],
    metadata_df: pd.DataFrame,
    top_n: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """Tìm 3-5 bài báo gần tâm cụm nhất để làm bài đại diện cho mỗi cụm."""
    labels = np.asarray(cluster_ids)
    distances = np.asarray(distances_to_center)

    if not (len(labels) == len(distances) == len(article_ids)):
        raise ValueError("Độ dài cluster_ids, distances và article_ids không khớp nhau")

    # Lập mapping thông tin từ metadata_df
    meta_lookup = {}
    if "article_id" in metadata_df.columns:
        for _, row in metadata_df.iterrows():
            aid = str(row["article_id"])
            meta_lookup[aid] = {
                "title": str(row.get("title", "")),
                "category_gold": str(row.get("category_gold", "")),
                "source": str(row.get("source", "")),
                "published_at": str(row.get("published_at", "")),
            }

    unique_clusters = np.unique(labels)
    representative_docs: dict[str, list[dict[str, Any]]] = {}

    for k in sorted(unique_clusters):
        cluster_mask = (labels == k)
        cluster_indices = np.where(cluster_mask)[0]

        # Sắp xếp theo khoảng cách đến tâm tăng dần (gần tâm nhất lên đầu)
        sorted_order = np.argsort(distances[cluster_indices])[:top_n]
        top_indices = cluster_indices[sorted_order]

        docs = []
        for rank, idx in enumerate(top_indices, start=1):
            aid = str(article_ids[idx])
            doc_meta = meta_lookup.get(aid, {})
            docs.append({
                "rank": rank,
                "article_id": aid,
                "title": doc_meta.get("title", ""),
                "category_gold": doc_meta.get("category_gold", ""),
                "source": doc_meta.get("source", ""),
                "published_at": doc_meta.get("published_at", ""),
                "distance_to_center": round(float(distances[idx]), 4),
            })
        representative_docs[str(k)] = docs

    return representative_docs


def save_kmeans_artifacts(
    output_dir: str | Path,
    model: KMeans,
    config: dict[str, Any],
    run_metadata: dict[str, Any],
    document_topics_df: pd.DataFrame,
    topic_keywords: dict[str, list[dict[str, Any]]],
    representative_docs: dict[str, list[dict[str, Any]]],
) -> None:
    """Lưu trữ đầy đủ bộ artifacts K-Means"""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Model checkpoint
    joblib.dump(model, out_path / "kmeans_model.joblib")

    # Config file
    with open(out_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # Run metadata
    with open(out_path / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, ensure_ascii=False, indent=2)

    # Document topics assignments (CSV)
    csv_path = out_path / "document_topics.csv"
    document_topics_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Topic keywords
    with open(out_path / "topic_keywords.json", "w", encoding="utf-8") as f:
        json.dump(topic_keywords, f, ensure_ascii=False, indent=2)

    # Representative documents
    with open(out_path / "representative_docs.json", "w", encoding="utf-8") as f:
        json.dump(representative_docs, f, ensure_ascii=False, indent=2)


def load_kmeans_artifacts(output_dir: str | Path) -> dict[str, Any]:
    """Tải và xác thực bộ artifacts K-Means."""
    dir_path = Path(output_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục artifacts: {dir_path}")

    model_path = dir_path / "kmeans_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model checkpoint: {model_path}")
    model: KMeans = joblib.load(model_path)

    config_path = dir_path / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = json.load(f)

    meta_path = dir_path / "run_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        run_metadata: dict[str, Any] = json.load(f)

    topics_path = dir_path / "document_topics.csv"
    if not topics_path.exists():
        raise FileNotFoundError(f"Không tìm thấy document_topics.csv: {topics_path}")
    document_topics = pd.read_csv(topics_path, encoding="utf-8-sig")

    keywords_path = dir_path / "topic_keywords.json"
    with open(keywords_path, "r", encoding="utf-8") as f:
        topic_keywords: dict[str, list[dict[str, Any]]] = json.load(f)

    rep_path = dir_path / "representative_docs.json"
    with open(rep_path, "r", encoding="utf-8") as f:
        representative_docs: dict[str, list[dict[str, Any]]] = json.load(f)

    return {
        "model": model,
        "config": config,
        "run_metadata": run_metadata,
        "document_topics": document_topics,
        "topic_keywords": topic_keywords,
        "representative_docs": representative_docs,
    }
