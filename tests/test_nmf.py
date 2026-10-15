"""Kiểm tra logic quan trọng của pipeline NMF M04-M06."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from experiments.nmf.metrics import (
    normalized_entropy,
    normalized_reconstruction_error,
    topic_diversity,
)
from experiments.nmf.run_experiment import choose_topic_count
from src.models.nmf import NMFConfig, NMFTopicModel, normalize_topic_weights


def test_normalize_topic_weights_preserves_zero_row() -> None:
    """Dòng không có trọng số phải giữ bằng 0 thay vì sinh NaN."""
    weights = np.array([[2.0, 1.0], [0.0, 0.0]])

    normalized = normalize_topic_weights(weights)

    np.testing.assert_allclose(normalized[0], [2 / 3, 1 / 3])
    np.testing.assert_array_equal(normalized[1], [0.0, 0.0])


def test_nmf_exports_document_and_topic_matrices() -> None:
    """M04 phải tạo được cả document-topic W và topic-word H."""
    matrix = sparse.csr_matrix(
        [
            [3.0, 2.0, 0.0, 0.0],
            [2.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 2.0],
            [0.0, 0.0, 2.0, 3.0],
        ]
    )
    model = NMFTopicModel(NMFConfig(n_topics=2, random_seed=42, max_iter=200))

    model.fit(matrix, ["kinh_te", "doanh_nghiep", "the_thao", "tran_dau"])

    assert model.document_topic_weights_.shape == (4, 2)
    assert model.topic_word_weights_.shape == (2, 4)
    np.testing.assert_allclose(model.training_topic_distributions().sum(axis=1), 1.0)


def test_reconstruction_error_uses_sparse_safe_formula() -> None:
    """Công thức metric phải khớp phép tính dense trên ví dụ nhỏ."""
    matrix = sparse.csr_matrix([[1.0, 0.0], [0.0, 2.0]])
    document_weights = np.array([[1.0], [2.0]])
    topic_weights = np.array([[1.0, 0.0]])
    expected = np.linalg.norm(matrix.toarray() - document_weights @ topic_weights)
    expected /= np.linalg.norm(matrix.toarray())

    actual = normalized_reconstruction_error(
        matrix, document_weights, topic_weights
    )

    assert actual == pytest.approx(expected)


def test_choose_topic_count_uses_declared_tie_breakers() -> None:
    """M05 ưu tiên NPMI rồi diversity, error và K theo đúng tài liệu."""
    metrics = pd.DataFrame(
        [
            {
                "n_topics": 4,
                "npmi_coherence": 0.20,
                "topic_diversity": 0.80,
                "normalized_reconstruction_error": 0.40,
            },
            {
                "n_topics": 6,
                "npmi_coherence": 0.25,
                "topic_diversity": 0.70,
                "normalized_reconstruction_error": 0.35,
            },
            {
                "n_topics": 8,
                "npmi_coherence": 0.25,
                "topic_diversity": 0.75,
                "normalized_reconstruction_error": 0.50,
            },
        ]
    )

    assert choose_topic_count(metrics) == 8


def test_multi_topic_entropy_is_higher_for_balanced_article() -> None:
    """M06 cần nhận diện bài phân bổ đều qua nhiều topic."""
    distributions = np.array([[0.50, 0.50], [0.99, 0.01]])

    entropy = normalized_entropy(distributions)

    assert entropy[0] > entropy[1]
    assert topic_diversity([["a", "b"], ["b", "c"]]) == pytest.approx(0.75)
