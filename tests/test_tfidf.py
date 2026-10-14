"""Kiểm thử đơn vị và tích hợp cho pipeline TF-IDF (Task M01)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse

# Hỗ trợ chạy trực tiếp bằng python tests/test_tfidf.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.tfidf import (
    create_tfidf_vectorizer,
    evaluate_validation_vocabulary,
    fit_tfidf_pipeline,
    load_stopwords,
    load_tfidf_artifacts,
    save_tfidf_artifacts,
    transform_tfidf_pipeline,
)


def test_stopwords_loading(tmp_path: Path) -> None:
    """Kiểm tra đọc stopwords, loại bỏ chú thích '#' và dòng trống."""
    stop_file = tmp_path / "test_stopwords.txt"
    stop_file.write_text("# Chú thích\n\nvà\ncủa\n# Dòng khác\ncho\n", encoding="utf-8")

    words = load_stopwords(stop_file)
    assert words == ["và", "của", "cho"]

    assert load_stopwords(None) is None

    with pytest.raises(FileNotFoundError):
        load_stopwords(tmp_path / "non_existent.txt")


def test_create_tfidf_vectorizer_config() -> None:
    """Kiểm tra khởi tạo TfidfVectorizer với cấu hình tùy chỉnh."""
    config = {
        "vectorizer": {
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.9,
            "max_features": 5000,
            "sublinear_tf": True,
            "norm": "l2",
            "token_pattern": r"(?u)\b\w+\b",
        }
    }
    vec = create_tfidf_vectorizer(config)
    assert vec.ngram_range == (1, 2)
    assert vec.min_df == 2
    assert vec.max_df == 0.9
    assert vec.max_features == 5000
    assert vec.sublinear_tf is True
    assert vec.norm == "l2"
    assert vec.token_pattern == r"(?u)\b\w+\b"


def test_fit_transform_no_data_leakage() -> None:
    """Kiểm tra fit duy nhất trên train; val có từ mới không làm biến đổi từ vựng."""
    train_texts = [
        "công_nghệ trí_tuệ_nhân_tạo phát_triển mạnh_mẽ",
        "khoa_học công_nghệ đóng_góp phát_triển kinh_tế",
        "kinh_tế số và công_nghệ thông_tin hiện_đại",
    ]
    val_texts = [
        "công_nghệ phát_triển nhanh_chóng",
        "y_tế đột_phá vũ_trụ sao_hỏa",  # Các từ hoàn toàn mới ngoài train
    ]

    config = {
        "vectorizer": {
            "ngram_range": [1, 1],
            "min_df": 1,
            "max_features": 100,
            "sublinear_tf": False,
        }
    }

    # Fit trên train
    vectorizer = fit_tfidf_pipeline(train_texts, config)
    train_vocab = set(vectorizer.get_feature_names_out())

    # Các từ mới từ val KHÔNG được xuất hiện trong vocabulary
    assert "y_tế" not in train_vocab
    assert "vũ_trụ" not in train_vocab
    assert "sao_hỏa" not in train_vocab

    # Transform train và val
    train_matrix = transform_tfidf_pipeline(vectorizer, train_texts)
    val_matrix = transform_tfidf_pipeline(vectorizer, val_texts)

    assert train_matrix.shape == (3, len(train_vocab))
    assert val_matrix.shape == (2, len(train_vocab))

    # Từ vựng của vectorizer không bị thay đổi sau transform
    assert set(vectorizer.get_feature_names_out()) == train_vocab


def test_output_sparse_matrix_properties() -> None:
    """Kiểm tra ma trận thưa hợp lệ: csr_matrix, chuẩn hóa l2, không chứa NaN."""
    train_texts = [
        "kinh_tế việt_nam tăng_trưởng mạnh",
        "xuất_khẩu gạo việt_nam đạt kỷ_lục",
        "thị_trường chứng_khoán tăng_trưởng",
    ]
    config = {
        "vectorizer": {
            "ngram_range": [1, 1],
            "min_df": 1,
            "norm": "l2",
        }
    }
    vectorizer = fit_tfidf_pipeline(train_texts, config)
    matrix = transform_tfidf_pipeline(vectorizer, train_texts)

    assert issparse(matrix)
    assert isinstance(matrix, csr_matrix)
    assert np.isfinite(matrix.data).all()
    assert (matrix.data >= 0.0).all()

    # Kiểm tra chuẩn hóa L2 của các hàng
    dense = matrix.toarray()
    row_norms = np.linalg.norm(dense, axis=1)
    np.testing.assert_allclose(row_norms, np.ones(3), rtol=1e-5)


def test_article_ids_alignment() -> None:
    """Kiểm tra số hàng ma trận khớp số article_ids."""
    texts = ["bài báo một", "bài báo hai", "bài báo ba"]
    article_ids = ["ART_001", "ART_002", "ART_003"]

    config = {"vectorizer": {"min_df": 1}}
    vectorizer = fit_tfidf_pipeline(texts, config)
    matrix = transform_tfidf_pipeline(vectorizer, texts)

    assert matrix.shape[0] == len(article_ids)


def test_evaluate_validation_vocabulary() -> None:
    """Kiểm tra tính toán OOV, độ thưa, active terms trên tập validation."""
    train_texts = ["tin thời_tiết hôm_nay mưa to gió lớn", "dự_báo thời_tiết ngày mai"]
    val_texts = [
        "thời_tiết hôm_nay nắng đẹp",
        "người ngoài hành_tinh khám_phá sao_hỏa",
    ]
    config = {"vectorizer": {"min_df": 1, "max_df": 1.0, "ngram_range": [1, 1]}}
    vectorizer = fit_tfidf_pipeline(train_texts, config)

    val_eval = evaluate_validation_vocabulary(vectorizer, val_texts)

    assert "vocab_size" in val_eval
    assert "val_num_docs" in val_eval
    assert val_eval["val_num_docs"] == 2
    assert "val_oov_rate_pct" in val_eval
    assert "val_mean_active_terms" in val_eval
    assert "val_zero_vector_docs" in val_eval
    assert "val_zero_vector_pct" in val_eval
    assert "val_sparsity_pct" in val_eval

    # Doc 2 hoàn toàn không có từ nào trong train_vocab => zero vector
    assert val_eval["val_zero_vector_docs"] == 1
    assert val_eval["val_zero_vector_pct"] == 50.0
    assert val_eval["val_oov_rate_pct"] > 0.0


def test_save_and_load_artifacts_roundtrip(tmp_path: Path) -> None:
    """Kiểm tra quy trình lưu và nạp lại artifacts chuẩn Handbook Mục E1."""
    train_texts = ["bài học tiếng việt", "học máy nlp cơ bản"]
    val_texts = ["tiếng việt nlp"]
    test_texts = ["học máy cơ bản"]

    train_ids = ["T1", "T2"]
    val_ids = ["V1"]
    test_ids = ["E1"]

    config = {
        "dataset": {"name": "toy_data"},
        "vectorizer": {"min_df": 1, "ngram_range": [1, 1]},
    }
    vectorizer = fit_tfidf_pipeline(train_texts, config)

    train_mat = transform_tfidf_pipeline(vectorizer, train_texts)
    val_mat = transform_tfidf_pipeline(vectorizer, val_texts)
    test_mat = transform_tfidf_pipeline(vectorizer, test_texts)

    run_metadata = {
        "pipeline": "M01_test",
        "created_at": "2026-10-14T00:00:00Z",
        "vocab_size": len(vectorizer.get_feature_names_out()),
    }

    out_dir = tmp_path / "artifacts" / "tfidf"
    save_tfidf_artifacts(
        output_dir=out_dir,
        vectorizer=vectorizer,
        config=config,
        run_metadata=run_metadata,
        splits_data={
            "train": (train_mat, train_ids),
            "val": (val_mat, val_ids),
            "test": (test_mat, test_ids),
        },
    )

    # Kiểm tra tồn tại đủ 10 files
    expected_files = [
        "tfidf_vectorizer.joblib",
        "feature_names.json",
        "config.json",
        "run_metadata.json",
        "tfidf_train.npz",
        "train_article_ids.json",
        "tfidf_val.npz",
        "val_article_ids.json",
        "tfidf_test.npz",
        "test_article_ids.json",
    ]
    for fn in expected_files:
        assert (out_dir / fn).exists(), f"Thiếu file: {fn}"

    # Tải lại split train
    loaded_train = load_tfidf_artifacts(out_dir, split="train")
    assert loaded_train["article_ids"] == train_ids
    assert loaded_train["feature_names"] == vectorizer.get_feature_names_out().tolist()
    assert loaded_train["config"] == config
    assert loaded_train["run_metadata"] == run_metadata
    np.testing.assert_allclose(
        loaded_train["matrix"].toarray(),
        train_mat.toarray(),
    )

    # Tải lại split val
    loaded_val = load_tfidf_artifacts(out_dir, split="val")
    assert loaded_val["article_ids"] == val_ids
    np.testing.assert_allclose(
        loaded_val["matrix"].toarray(),
        val_mat.toarray(),
    )


def test_error_handling(tmp_path: Path) -> None:
    """Kiểm tra xử lý ngoại lệ cho các trường hợp dữ liệu bất thường."""
    # Danh sách văn bản rỗng
    with pytest.raises(ValueError, match="Danh sách văn bản huấn luyện rỗng"):
        fit_tfidf_pipeline([], {"vectorizer": {}})

    with pytest.raises(ValueError, match="Danh sách văn bản biến đổi rỗng"):
        transform_tfidf_pipeline(create_tfidf_vectorizer({}), [])

    # Lệch kích thước giữa ma trận và article_ids khi lưu
    fitted_vec = fit_tfidf_pipeline(["bài một", "bài hai"], {"vectorizer": {"min_df": 1, "max_df": 1.0}})
    mat = csr_matrix(np.ones((3, len(fitted_vec.get_feature_names_out()))))
    ids = ["ID1", "ID2"]  # Chỉ có 2 IDs trong khi ma trận có 3 hàng
    with pytest.raises(ValueError, match="không khớp"):
        save_tfidf_artifacts(
            tmp_path / "err_out",
            fitted_vec,
            {},
            {},
            {"train": (mat, ids)},
        )

    # Nạp từ thư mục không tồn tại
    with pytest.raises(FileNotFoundError):
        load_tfidf_artifacts(tmp_path / "non_existent_dir", split="train")


def test_official_artifacts_exist_and_conform_to_handbook() -> None:
    """Kiểm tra bộ artifacts chính thức trong artifacts/tfidf/ tuân thủ Handbook."""
    artifacts_dir = Path("artifacts/tfidf")
    if not artifacts_dir.exists():
        pytest.skip("Chưa tạo artifacts/tfidf chính thức")

    expected_files = [
        "tfidf_vectorizer.joblib",
        "feature_names.json",
        "config.json",
        "run_metadata.json",
        "tfidf_train.npz",
        "train_article_ids.json",
        "tfidf_val.npz",
        "val_article_ids.json",
        "tfidf_test.npz",
        "test_article_ids.json",
    ]
    for fname in expected_files:
        fpath = artifacts_dir / fname
        assert fpath.exists(), f"Thiếu artifact chính thức: {fname}"

    # Đọc và kiểm tra kích thước
    train_loaded = load_tfidf_artifacts(artifacts_dir, split="train")
    val_loaded = load_tfidf_artifacts(artifacts_dir, split="val")
    test_loaded = load_tfidf_artifacts(artifacts_dir, split="test")

    # Số dòng tương ứng: Train=6493, Val=1391, Test=1390
    assert train_loaded["matrix"].shape[0] == 6493
    assert len(train_loaded["article_ids"]) == 6493

    assert val_loaded["matrix"].shape[0] == 1391
    assert len(val_loaded["article_ids"]) == 1391

    assert test_loaded["matrix"].shape[0] == 1390
    assert len(test_loaded["article_ids"]) == 1390

    # Số cột bằng nhau ở cả 3 tập
    n_features = train_loaded["matrix"].shape[1]
    assert n_features == val_loaded["matrix"].shape[1]
    assert n_features == test_loaded["matrix"].shape[1]
    assert n_features == len(train_loaded["feature_names"])
    assert n_features == 20000

    # Không có giá trị vô lý
    assert np.isfinite(train_loaded["matrix"].data).all()
    assert np.isfinite(val_loaded["matrix"].data).all()
    assert np.isfinite(test_loaded["matrix"].data).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
