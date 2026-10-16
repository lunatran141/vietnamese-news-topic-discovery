
"""Bag-of-Words feature extraction for M07 LDA."""

from collections.abc import Sequence

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer


def create_bow_vectorizer(config: dict) -> CountVectorizer:
    """Tạo CountVectorizer từ cấu hình YAML."""
    return CountVectorizer(
        max_features=config.get("max_features", 15000),
        min_df=config.get("min_df", 3),
        max_df=config.get("max_df", 0.85),
        ngram_range=tuple(config.get("ngram_range", [1, 1])),
        token_pattern=config.get(
            "token_pattern", r"(?u)\b\w\w+\b"
        ),
        lowercase=False,
        dtype=np.int64,
    )


def fit_bow_pipeline(
    train_texts: Sequence[str],
    config: dict,
) -> tuple[CountVectorizer, csr_matrix]:
    """Fit từ vựng chỉ trên Train và tạo ma trận BoW."""
    vectorizer = create_bow_vectorizer(config)
    train_matrix = vectorizer.fit_transform(train_texts)

    if train_matrix.shape[1] == 0:
        raise ValueError("BoW vocabulary rỗng")

    return vectorizer, train_matrix.tocsr()


def transform_bow_pipeline(
    vectorizer: CountVectorizer,
    texts: Sequence[str],
) -> csr_matrix:
    """Biến đổi văn bản bằng vocabulary đã học trên Train."""
    matrix = vectorizer.transform(texts).tocsr()

    if matrix.shape[1] != len(vectorizer.vocabulary_):
        raise ValueError("BoW vocabulary không nhất quán")

    return matrix
