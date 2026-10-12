import numpy as np
import pandas as pd
import pytest

from src.models.hdbscan import cluster_articles

class FakeClusterer:
    """Kết quả giả cố định để kiểm tra ghép ID và kiểm tra đầu vào."""

    def fit_predict(self, vectors):
        self.received_vectors = vectors.copy()
        self.probabilities_ = np.array([0.9, 0.0, 0.8])
        return np.array([2, -1, 2])

@pytest.fixture
def embeddings():
    return pd.DataFrame({
        "article_id": ["bai_C", "bai_A", "bai_B"],
        "embedding": [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.5, 0.5]),
        ],
        "model_name": "test-model",
        "data_version": "test-v1",
    })

def test_giu_id_va_khong_chay_umap_lai(embeddings, monkeypatch):
    def forbid_umap(*args, **kwargs):
        raise AssertionError("Không được chạy UMAP khi đã có reduced_vectors")

    monkeypatch.setattr("src.models.hdbscan.UMAP", forbid_umap)

    reduced = np.array([[3.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    clusterer = FakeClusterer()
    result = cluster_articles(
        embeddings,
        reduced_vectors=reduced,
        clusterer=clusterer,
    )

    assert result["article_id"].tolist() == ["bai_C", "bai_A", "bai_B"]
    assert result["cluster_id"].tolist() == [2, -1, 2]
    assert result["membership_strength"].tolist() == [0.9, 0.0, 0.8]
    assert result["model_name"].eq("test-model").all()
    assert result["data_version"].eq("test-v1").all()
    np.testing.assert_array_equal(clusterer.received_vectors, reduced)


def test_tu_choi_ma_tran_giam_chieu_sai_so_bai(embeddings):
    with pytest.raises(ValueError, match="không khớp số bài"):
        cluster_articles(
            embeddings,
            reduced_vectors=np.zeros((2, 2)),
            clusterer=FakeClusterer(),
        )

def test_tu_choi_embedding_nan(embeddings):
    embeddings.at[0, "embedding"] = np.array([np.nan, 0.0])
    with pytest.raises(ValueError, match="NaN"):
        cluster_articles(
            embeddings,
            reduced_vectors=np.zeros((3, 2)),
            clusterer=FakeClusterer(),
        )

def test_tu_choi_membership_nan(embeddings):
    class BadClusterer:
        def fit_predict(self, vectors):
            self.probabilities_ = np.array([0.9, np.nan, 0.8])
            return np.array([2, -1, 2])

    with pytest.raises(ValueError, match="Membership"):
        cluster_articles(
            embeddings,
            reduced_vectors=np.zeros((3, 2)),
            clusterer=BadClusterer(),
        )