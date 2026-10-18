import numpy as np
import pandas as pd
import pytest

from experiments.hdbscan.run_m11_validation import (
    verify_train_reproduction,
)
from src.models.hdbscan import predict_hdbscan_validation


def test_tu_choi_model_co_nhan_train_khac_m11():
    saved = pd.DataFrame({
        "article_id": ["a", "b"],
        "experiment": ["eom_15_5", "eom_15_5"],
        "cluster_id": [0, -1],
        "config_hash": ["hash-1", "hash-1"],
    })

    with pytest.raises(ValueError, match="nhãn train khác"):
        verify_train_reproduction(
            ["a", "b"],
            np.array([0, 1]),
            saved,
            "eom_15_5",
            "hash-1",
        )


def test_validation_chi_transform_va_predict_khong_fit_lai(monkeypatch):
    table = pd.DataFrame({
        "article_id": ["val-b", "val-a"],
        "embedding": [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ],
    })

    class FakeReducer:
        def transform(self, vectors):
            np.testing.assert_array_equal(
                vectors,
                np.array([[1.0, 0.0], [0.0, 1.0]]),
            )
            return vectors + 2

        def fit(self, _):
            raise AssertionError("Không được fit trên validation")

        def fit_transform(self, _):
            raise AssertionError("Không được fit trên validation")

    class FakeClusterer:
        def fit(self, _):
            raise AssertionError("Không được fit trên validation")

        def fit_predict(self, _):
            raise AssertionError("Không được fit trên validation")

    def fake_approximate_predict(clusterer, vectors):
        assert isinstance(clusterer, FakeClusterer)
        np.testing.assert_array_equal(
            vectors,
            np.array([[3.0, 2.0], [2.0, 3.0]]),
        )
        return np.array([4, -1]), np.array([0.8, 0.0])

    monkeypatch.setattr(
        "src.models.hdbscan.approximate_predict",
        fake_approximate_predict,
    )

    result = predict_hdbscan_validation(
        table,
        FakeReducer(),
        FakeClusterer(),
    )

    assert result["article_id"].tolist() == ["val-b", "val-a"]
    assert result["cluster_id"].tolist() == [4, -1]
    assert result["membership_strength"].tolist() == [0.8, 0.0]