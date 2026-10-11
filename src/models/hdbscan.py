import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from umap import UMAP


def cluster_articles(
    embedding_table: pd.DataFrame,
    reducer=None,
    clusterer=None,
) -> pd.DataFrame:
    """Giảm chiều vector, phân cụm và ghép nhãn với article_id."""

    required = {"article_id", "embedding"}
    missing = required - set(embedding_table.columns)
    if missing:
        raise ValueError(f"Thiếu cột: {sorted(missing)}")

    if embedding_table.empty:
        raise ValueError("Bảng embedding rỗng")

    if embedding_table["article_id"].isna().any():
        raise ValueError("Có article_id bị thiếu")

    if embedding_table["article_id"].duplicated().any():
        raise ValueError("Có article_id trùng")

    try:
        vectors = np.stack(
            [np.asarray(v, dtype=np.float32)
             for v in embedding_table["embedding"]]
        )
    except ValueError as exc:
        raise ValueError("Các vector không cùng số chiều") from exc

    if vectors.ndim != 2:
        raise ValueError("Embedding phải là ma trận 2 chiều")

    if not np.isfinite(vectors).all():
        raise ValueError("Embedding chứa NaN hoặc Infinity")

    if reducer is None:
        if len(vectors) < 20:
            raise ValueError("Cần ít nhất 20 bài cho cấu hình UMAP mặc định")

        reducer = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )

    if clusterer is None:
        clusterer = HDBSCAN(
            min_cluster_size=5,
            min_samples=2,
            metric="euclidean",
        )

    reduced_vectors = reducer.fit_transform(vectors)
    labels = np.asarray(clusterer.fit_predict(reduced_vectors))
    strengths = np.asarray(clusterer.probabilities_)

    if len(labels) != len(embedding_table):
        raise ValueError("Số nhãn không khớp số bài báo")

    if len(strengths) != len(embedding_table):
        raise ValueError("Số điểm membership không khớp số bài báo")

    result = pd.DataFrame({
        "article_id": embedding_table["article_id"].tolist(),
        "cluster_id": labels.astype(int),
        "membership_strength": strengths.astype(float),
    })

    for column in ("model_name", "data_version"):
        if column in embedding_table.columns:
            result[column] = embedding_table[column].tolist()

    return result