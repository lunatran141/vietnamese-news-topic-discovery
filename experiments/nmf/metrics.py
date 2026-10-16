"""Các metric phục vụ chọn số topic và đánh giá NMF."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import sparse


def topic_diversity(topic_terms: Sequence[Sequence[str]]) -> float:
    """Tính tỷ lệ từ khóa duy nhất trên toàn bộ vị trí top words."""
    flattened = [term for terms in topic_terms for term in terms]
    return len(set(flattened)) / len(flattened) if flattened else 0.0


def npmi_coherence(
    document_term_matrix,
    topic_feature_ids: Sequence[Sequence[int]],
) -> tuple[float, list[float]]:
    """Tính NPMI từ đồng xuất hiện của top words ở cấp document.

    Ma trận TF-IDF chỉ được dùng để xác định một từ có xuất hiện hay không.
    Các cặp chưa từng đồng xuất hiện nhận điểm -1 theo định nghĩa NPMI.
    """
    document_count = int(document_term_matrix.shape[0])
    if document_count == 0:
        raise ValueError("Không thể tính coherence trên tập rỗng")

    unique_ids = sorted(
        {int(feature_id) for topic in topic_feature_ids for feature_id in topic}
    )
    if len(unique_ids) < 2:
        return 0.0, [0.0 for _ in topic_feature_ids]

    presence = document_term_matrix[:, unique_ids].tocsr(copy=True)
    presence.data[:] = 1.0
    cooccurrence = (presence.T @ presence).toarray().astype(np.float64)
    document_frequency = np.asarray(presence.sum(axis=0)).ravel().astype(np.float64)
    local_index = {feature_id: index for index, feature_id in enumerate(unique_ids)}

    topic_scores: list[float] = []
    for feature_ids in topic_feature_ids:
        pair_scores: list[float] = []
        valid_ids = [int(value) for value in feature_ids if int(value) in local_index]
        for left_position, left_id in enumerate(valid_ids):
            for right_id in valid_ids[left_position + 1 :]:
                left = local_index[left_id]
                right = local_index[right_id]
                pair_probability = cooccurrence[left, right] / document_count
                if pair_probability <= 0:
                    pair_scores.append(-1.0)
                    continue

                left_probability = document_frequency[left] / document_count
                right_probability = document_frequency[right] / document_count
                denominator = -np.log(pair_probability)
                if denominator <= np.finfo(float).eps:
                    pair_scores.append(1.0)
                    continue
                pair_scores.append(
                    float(
                        np.log(
                            pair_probability
                            / (left_probability * right_probability)
                        )
                        / denominator
                    )
                )
        topic_scores.append(float(np.mean(pair_scores)) if pair_scores else 0.0)

    overall_score = float(np.mean(topic_scores)) if topic_scores else 0.0
    return overall_score, topic_scores


def normalized_reconstruction_error(
    document_term_matrix,
    document_topic_weights: np.ndarray,
    topic_word_weights: np.ndarray,
) -> float:
    """Tính ||X - WH||_F / ||X||_F cho một split bất kỳ."""
    matrix_norm_squared = float(sparse.linalg.norm(document_term_matrix) ** 2)
    if matrix_norm_squared == 0:
        return 0.0

    # Dùng đẳng thức Frobenius để tránh tạo ma trận WH đặc rất lớn.
    projected_input = document_term_matrix @ topic_word_weights.T
    cross_term = float(np.sum(document_topic_weights * projected_input))
    weight_gram = document_topic_weights.T @ document_topic_weights
    topic_gram = topic_word_weights @ topic_word_weights.T
    reconstruction_norm_squared = float(np.sum(weight_gram * topic_gram))
    residual_squared = max(
        matrix_norm_squared - (2.0 * cross_term) + reconstruction_norm_squared,
        0.0,
    )
    return float(np.sqrt(residual_squared / matrix_norm_squared))


def normalized_entropy(topic_distributions: np.ndarray) -> np.ndarray:
    """Đo độ phân tán topic của từng bài trên thang 0-1."""
    values = np.asarray(topic_distributions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("Cần ít nhất hai topic để tính entropy chuẩn hóa")
    safe_values = np.clip(values, np.finfo(float).eps, 1.0)
    entropy = -np.sum(safe_values * np.log(safe_values), axis=1)
    return entropy / np.log(values.shape[1])
