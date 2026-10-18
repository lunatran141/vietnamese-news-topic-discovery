"""Các hàm giảm chiều và phân cụm dùng chung cho HDBSCAN."""

import numpy as np
import pandas as pd
from hdbscan import HDBSCAN, approximate_predict
from umap import UMAP

def _get_vectors(embedding_table: pd.DataFrame) -> np.ndarray:
    """Kiểm tra bảng embedding và chuyển vector thành ma trận."""
    required = {"article_id", "embedding"}
    missing = required - set(embedding_table.columns)
    if missing:
        raise ValueError(f"Thiếu cột: {sorted(missing)}")

    if embedding_table.empty:
        raise ValueError("Bảng embedding rỗng")

    ids = embedding_table["article_id"]
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any():
        raise ValueError("Có article_id bị thiếu")
    if ids.duplicated().any():
        raise ValueError("Có article_id trùng")

    try:
        vectors = np.stack([
            np.asarray(vector, dtype=np.float32)
            for vector in embedding_table["embedding"]
        ])
    except (ValueError, TypeError) as exc:
        raise ValueError("Các vector không hợp lệ hoặc không cùng số chiều") from exc

    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise ValueError("Embedding phải là ma trận 2 chiều có số chiều > 0")
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding chứa NaN hoặc Infinity")

    return vectors


def reduce_embeddings(
    embedding_table: pd.DataFrame,
    umap_params: dict | None = None,
) -> np.ndarray:
    """Chạy UMAP, giữ nguyên thứ tự bài trong embedding_table."""
    vectors = _get_vectors(embedding_table)

    params = {
        "n_neighbors": 15,
        "n_components": 5,
        "min_dist": 0.0,
        "metric": "cosine",
        "random_state": 42,
    }
    if umap_params is not None:
        params.update(umap_params)

    return UMAP(**params).fit_transform(vectors)


def cluster_articles(
    embedding_table: pd.DataFrame,
    reducer=None,
    clusterer=None,
    *,
    reduced_vectors=None,
    hdbscan_params: dict | None = None,
) -> pd.DataFrame:
    """
    Phân cụm và ghép nhãn với article_id.

    Nếu truyền reduced_vectors, dùng trực tiếp kết quả giảm chiều
    và không chạy UMAP lại. Ma trận này phải giữ cùng thứ tự bài
    với embedding_table.
    """
    vectors = _get_vectors(embedding_table)

    if reduced_vectors is not None:
        if reducer is not None:
            raise ValueError("Chỉ truyền reducer hoặc reduced_vectors")
        reduced = np.asarray(reduced_vectors)
    elif reducer is not None:
        reduced = np.asarray(reducer.fit_transform(vectors))
    else:
        reduced = reduce_embeddings(embedding_table)

    if (
        reduced.ndim != 2
        or reduced.shape[0] != len(embedding_table)
        or reduced.shape[1] == 0
    ):
        raise ValueError("Ma trận giảm chiều không khớp số bài báo")
    if not np.isfinite(reduced).all():
        raise ValueError("Ma trận giảm chiều chứa NaN hoặc Infinity")

    if clusterer is not None and hdbscan_params is not None:
        raise ValueError("Chỉ truyền clusterer hoặc hdbscan_params")

    if clusterer is None:
        params = {
            "min_cluster_size": 5,
            "min_samples": 2,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
        }
        if hdbscan_params is not None:
            params.update(hdbscan_params)
        clusterer = HDBSCAN(**params)

    labels = np.asarray(clusterer.fit_predict(reduced), dtype=float)
    strengths = np.asarray(clusterer.probabilities_, dtype=float)

    if labels.shape != (len(embedding_table),):
        raise ValueError("Số nhãn không khớp số bài báo")
    if strengths.shape != (len(embedding_table),):
        raise ValueError("Số điểm membership không khớp số bài báo")

    if (
        not np.isfinite(labels).all()
        or np.any(labels < -1)
        or np.any(labels != np.floor(labels))
    ):
        raise ValueError("Nhãn cụm không hợp lệ")

    if (
        not np.isfinite(strengths).all()
        or np.any(strengths < 0)
        or np.any(strengths > 1)
    ):
        raise ValueError("Membership không hợp lệ")

    result = pd.DataFrame({
        "article_id": embedding_table["article_id"].tolist(),
        "cluster_id": labels.astype(int),
        "membership_strength": strengths,
    })

    for column in ("model_name", "data_version"):
        if column in embedding_table.columns:
            result[column] = embedding_table[column].tolist()

    return result

def fit_umap_for_validation(
    embedding_table: pd.DataFrame,
    umap_params: dict,
) -> tuple[UMAP, np.ndarray]:
    """Fit UMAP trên train và giữ model để transform validation."""
    vectors = _get_vectors(embedding_table)
    reducer = UMAP(**umap_params)
    reduced = np.asarray(reducer.fit_transform(vectors))

    if reduced.shape != (len(embedding_table), umap_params["n_components"]):
        raise ValueError("Kích thước UMAP train không hợp lệ")
    if not np.isfinite(reduced).all():
        raise ValueError("UMAP train chứa NaN hoặc Infinity")

    return reducer, reduced


def fit_hdbscan_for_validation(
    train_reduced: np.ndarray,
    hdbscan_params: dict,
) -> tuple[HDBSCAN, np.ndarray]:
    """Fit HDBSCAN trên train và giữ dữ liệu để dự đoán bài mới."""
    reduced = np.asarray(train_reduced)

    if reduced.ndim != 2 or len(reduced) == 0:
        raise ValueError("Ma trận UMAP train không hợp lệ")
    if not np.isfinite(reduced).all():
        raise ValueError("Ma trận UMAP train chứa NaN hoặc Infinity")

    params = dict(hdbscan_params)
    params["prediction_data"] = True

    clusterer = HDBSCAN(**params)
    labels = np.asarray(clusterer.fit_predict(reduced), dtype=int)

    if labels.shape != (len(reduced),):
        raise ValueError("Số nhãn train không khớp số bài")

    return clusterer, labels


def predict_hdbscan_validation(
    embedding_table: pd.DataFrame,
    reducer: UMAP,
    clusterer: HDBSCAN,
) -> pd.DataFrame:
    """Gán bài validation vào các cụm train đã có; không fit lại."""
    vectors = _get_vectors(embedding_table)
    reduced = np.asarray(reducer.transform(vectors))

    if reduced.ndim != 2 or reduced.shape[0] != len(embedding_table):
        raise ValueError("Số vector UMAP validation không khớp số bài")
    if not np.isfinite(reduced).all():
        raise ValueError("UMAP validation chứa NaN hoặc Infinity")

    labels, strengths = approximate_predict(clusterer, reduced)
    labels = np.asarray(labels, dtype=int)
    strengths = np.asarray(strengths, dtype=float)

    if labels.shape != (len(embedding_table),):
        raise ValueError("Số nhãn validation không khớp số bài")
    if strengths.shape != (len(embedding_table),):
        raise ValueError("Số điểm membership validation không khớp số bài")
    if np.any(labels < -1):
        raise ValueError("Nhãn validation không hợp lệ")
    if not np.isfinite(strengths).all():
        raise ValueError("Membership validation chứa NaN hoặc Infinity")
    if np.any((strengths < 0) | (strengths > 1)):
        raise ValueError("Membership validation nằm ngoài [0, 1]")

    return pd.DataFrame({
        "article_id": embedding_table["article_id"].tolist(),
        "cluster_id": labels,
        "membership_strength": strengths,
    })