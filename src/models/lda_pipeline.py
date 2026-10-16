
"""LDA topic modeling pipeline for Task M07."""

from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation


def create_lda_model(config: dict) -> LatentDirichletAllocation:
    """Khởi tạo LDA từ cấu hình YAML."""
    return LatentDirichletAllocation(
        n_components=config.get("n_components", 10),
        learning_method=config.get("learning_method", "batch"),
        max_iter=config.get("max_iter", 20),
        random_state=config.get("random_state", 42),
        evaluate_every=config.get("evaluate_every", -1),
        n_jobs=config.get("n_jobs", -1),
    )


def validate_bow_matrix(matrix: csr_matrix) -> None:
    """Kiểm tra ma trận đếm từ trước khi sử dụng."""
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError("BoW matrix rỗng hoặc sai kích thước")

    if not np.isfinite(matrix.data).all():
        raise ValueError("BoW chứa NaN hoặc Infinity")

    if (matrix.data < 0).any():
        raise ValueError("BoW chứa giá trị âm")

    if not np.all(matrix.data == np.floor(matrix.data)):
        raise ValueError("LDA yêu cầu số đếm từ nguyên")


def fit_lda_pipeline(
    train_matrix: csr_matrix,
    config: dict,
) -> LatentDirichletAllocation:
    """Huấn luyện LDA chỉ trên tập Train."""
    validate_bow_matrix(train_matrix)

    if train_matrix.nnz == 0:
        raise ValueError("Train BoW không có từ hợp lệ")

    model = create_lda_model(config)
    model.fit(train_matrix)
    return model


def transform_lda_pipeline(
    model: LatentDirichletAllocation,
    matrix: csr_matrix,
) -> np.ndarray:
    """Suy luận phân phối topic cho từng bài báo."""
    validate_bow_matrix(matrix)

    distributions = model.transform(matrix)

    if not np.isfinite(distributions).all():
        raise ValueError("Topic distributions không hợp lệ")

    if not np.allclose(
        distributions.sum(axis=1), 1.0, atol=1e-6
    ):
        raise ValueError("Tổng xác suất topic phải bằng 1")

    return distributions


def save_lda_model(
    model: LatentDirichletAllocation,
    output_path: str | Path,
) -> None:
    """Lưu mô hình đã huấn luyện."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_lda_model(
    model_path: str | Path,
) -> LatentDirichletAllocation:
    """Tải mô hình LDA từ file tin cậy."""
    path = Path(model_path)

    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy model: {path}")

    return joblib.load(path)
