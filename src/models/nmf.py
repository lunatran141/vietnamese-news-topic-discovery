"""Mô hình NMF dùng chung cho pipeline khám phá topic của TV2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import sparse
from sklearn.decomposition import NMF


@dataclass(frozen=True)
class NMFConfig:
    """Lưu các tham số quyết định một lần huấn luyện NMF.

    Args:
        n_topics: Số topic cần phân rã.
        random_seed: Seed giúp kết quả có thể tái lập.
        init: Phương pháp khởi tạo ma trận không âm.
        solver: Thuật toán tối ưu của scikit-learn.
        beta_loss: Hàm mất mát dùng khi tối ưu.
        max_iter: Số vòng lặp tối đa.
        tolerance: Ngưỡng hội tụ.
    """

    n_topics: int
    random_seed: int = 42
    init: str = "nndsvda"
    solver: str = "cd"
    beta_loss: str = "frobenius"
    max_iter: int = 500
    tolerance: float = 1e-4


def normalize_topic_weights(weights: np.ndarray) -> np.ndarray:
    """Chuẩn hóa trọng số topic của mỗi bài thành tổng bằng 1.

    Args:
        weights: Ma trận document-topic không âm.

    Returns:
        Ma trận cùng kích thước, mỗi dòng có tổng bằng 1 nếu dòng khác 0.

    Raises:
        ValueError: Khi đầu vào không phải ma trận hai chiều hoặc có số âm.
    """
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Ma trận document-topic phải có hai chiều")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Trọng số topic phải hữu hạn và không âm")

    row_sums = values.sum(axis=1, keepdims=True)
    return np.divide(
        values,
        row_sums,
        out=np.zeros_like(values),
        where=row_sums > 0,
    )


class NMFTopicModel:
    """Đóng gói NMF và giữ rõ hai ma trận W/H cho M04.

    Sau khi fit, ``document_topic_weights_`` là W và
    ``topic_word_weights_`` là H trong phép xấp xỉ X ≈ W @ H.
    """

    def __init__(self, config: NMFConfig) -> None:
        if config.n_topics < 2:
            raise ValueError("n_topics phải lớn hơn hoặc bằng 2")
        if config.max_iter < 1:
            raise ValueError("max_iter phải là số dương")
        self.config = config
        self.estimator_: NMF | None = None
        self.feature_names_: np.ndarray | None = None
        self.document_topic_weights_: np.ndarray | None = None
        self.topic_word_weights_: np.ndarray | None = None

    def fit(self, matrix, feature_names: Sequence[str]) -> "NMFTopicModel":
        """Huấn luyện NMF trên ma trận document-term không âm.

        Args:
            matrix: Ma trận sparse/dense có dạng (documents, terms).
            feature_names: Tên feature theo đúng thứ tự cột của ``matrix``.

        Returns:
            Chính đối tượng đã được huấn luyện.
        """
        self._validate_matrix(matrix, feature_names)
        self.estimator_ = NMF(
            n_components=self.config.n_topics,
            init=self.config.init,
            solver=self.config.solver,
            beta_loss=self.config.beta_loss,
            max_iter=self.config.max_iter,
            random_state=self.config.random_seed,
            tol=self.config.tolerance,
        )
        self.document_topic_weights_ = self.estimator_.fit_transform(matrix)
        self.topic_word_weights_ = self.estimator_.components_
        self.feature_names_ = np.asarray(feature_names, dtype=object)
        return self

    def transform(self, matrix) -> np.ndarray:
        """Suy diễn phân phối topic cho các bài không dùng để fit."""
        if self.estimator_ is None:
            raise RuntimeError("Cần fit mô hình trước khi transform")
        weights = self.estimator_.transform(matrix)
        return normalize_topic_weights(weights)

    def training_topic_distributions(self) -> np.ndarray:
        """Trả về W đã chuẩn hóa của tập dùng để huấn luyện."""
        if self.document_topic_weights_ is None:
            raise RuntimeError("Cần fit mô hình trước khi lấy phân phối topic")
        return normalize_topic_weights(self.document_topic_weights_)

    def top_terms(self, topic_index: int, top_n: int = 10) -> list[tuple[str, float]]:
        """Lấy từ khóa và trọng số cao nhất của một topic."""
        if self.topic_word_weights_ is None or self.feature_names_ is None:
            raise RuntimeError("Cần fit mô hình trước khi lấy từ khóa")
        if not 0 <= topic_index < self.config.n_topics:
            raise IndexError(f"Topic không tồn tại: {topic_index}")
        if top_n < 1:
            raise ValueError("top_n phải là số dương")

        weights = self.topic_word_weights_[topic_index]
        indexes = np.argsort(-weights, kind="stable")[:top_n]
        return [
            (str(self.feature_names_[index]), float(weights[index]))
            for index in indexes
        ]

    def _validate_matrix(self, matrix, feature_names: Sequence[str]) -> None:
        """Kiểm tra điều kiện đầu vào trước khi gọi scikit-learn."""
        if getattr(matrix, "ndim", None) != 2:
            raise ValueError("Ma trận document-term phải có hai chiều")
        if min(matrix.shape) < self.config.n_topics:
            raise ValueError("Hai chiều của ma trận phải không nhỏ hơn n_topics")
        if len(feature_names) != matrix.shape[1]:
            raise ValueError("Số feature name không khớp số cột của ma trận")
        nonzero_count = matrix.nnz if sparse.issparse(matrix) else np.count_nonzero(matrix)
        if nonzero_count == 0:
            raise ValueError("Ma trận document-term không được rỗng")

        values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("NMF chỉ nhận ma trận hữu hạn và không âm")
