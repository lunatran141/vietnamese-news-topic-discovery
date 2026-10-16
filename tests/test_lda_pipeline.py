
"""Unit tests for M07 Bag-of-Words and LDA pipeline."""

import numpy as np
import pytest

from src.features.bow import (
    fit_bow_pipeline,
    transform_bow_pipeline,
)
from src.models.lda_pipeline import (
    fit_lda_pipeline,
    transform_lda_pipeline,
    save_lda_model,
    load_lda_model,
)


@pytest.fixture
def sample_data():
    documents = [
        "kinh_tế thị_trường doanh_nghiệp đầu_tư",
        "doanh_nghiệp tài_chính ngân_hàng kinh_tế",
        "thị_trường kinh_tế tài_chính đầu_tư",
        "giáo_dục học_sinh trường_học giáo_viên",
        "giáo_viên đào_tạo học_sinh trường_học",
        "giáo_dục trường_học học_sinh đào_tạo",
    ]

    vectorizer, matrix = fit_bow_pipeline(
        documents,
        {"min_df": 1, "max_df": 1.0},
    )

    return vectorizer, matrix


def test_bow_matrix(sample_data):
    vectorizer, matrix = sample_data

    assert matrix.shape[0] == 6
    assert matrix.shape[1] == len(
        vectorizer.get_feature_names_out()
    )
    assert np.all(matrix.data >= 0)
    assert np.all(matrix.data == np.floor(matrix.data))


def test_bow_vocabulary_does_not_change(sample_data):
    vectorizer, _ = sample_data
    original_vocabulary = vectorizer.vocabulary_.copy()

    matrix = transform_bow_pipeline(
        vectorizer,
        ["kinh_tế từ_mới_chưa_từng_xuất_hiện"],
    )

    assert vectorizer.vocabulary_ == original_vocabulary
    assert matrix.shape[1] == len(original_vocabulary)
    assert "từ_mới_chưa_từng_xuất_hiện" not in original_vocabulary


def test_lda_topic_distributions(sample_data):
    _, matrix = sample_data

    model = fit_lda_pipeline(
        matrix,
        {
            "n_components": 2,
            "max_iter": 20,
            "random_state": 42,
            "n_jobs": 1,
        },
    )

    distributions = transform_lda_pipeline(model, matrix)

    assert distributions.shape == (6, 2)
    assert np.isfinite(distributions).all()
    assert np.all(distributions >= 0)
    assert np.allclose(
        distributions.sum(axis=1),
        1.0,
        atol=1e-6,
    )


def test_lda_reproducibility(sample_data):
    _, matrix = sample_data

    config = {
        "n_components": 2,
        "max_iter": 20,
        "random_state": 42,
        "n_jobs": 1,
    }

    model_1 = fit_lda_pipeline(matrix, config)
    model_2 = fit_lda_pipeline(matrix, config)

    np.testing.assert_allclose(
        model_1.components_,
        model_2.components_,
        rtol=1e-6,
        atol=1e-8,
    )


def test_lda_model_save_load(sample_data, tmp_path):
    _, matrix = sample_data

    model = fit_lda_pipeline(
        matrix,
        {
            "n_components": 2,
            "random_state": 42,
            "n_jobs": 1,
        },
    )

    model_path = tmp_path / "lda_model.joblib"
    save_lda_model(model, model_path)

    loaded_model = load_lda_model(model_path)

    np.testing.assert_allclose(
        model.transform(matrix),
        loaded_model.transform(matrix),
    )


def test_lda_rejects_negative_counts(sample_data):
    _, matrix = sample_data
    invalid_matrix = matrix.copy().astype(float)
    invalid_matrix.data[0] = -1.0

    with pytest.raises(ValueError):
        fit_lda_pipeline(
            invalid_matrix,
            {"n_components": 2},
        )
