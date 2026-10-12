import numpy as np
import pandas as pd
import pytest

from src.features.embeddings import (
    create_embeddings,
    load_embedding_artifact,
    save_embedding_artifact,
)


class FakeEncoder:
    """Mô hình giả để test mà không cần tải mô hình thật."""

    def __init__(self):
        self.received_texts = None

    def encode(self, texts, **kwargs):
        self.received_texts = texts
        return np.array(
            [[float(i), float(i + 1)] for i in range(len(texts))],
            dtype=np.float32,
        )


def test_so_vector_so_chieu_va_thu_tu_article_id():
    articles = pd.DataFrame({
        "article_id": ["BAI_B", "BAI_A", "BAI_C"],
        "title": ["Tin B", "Tin A", "Tin C"],
        "content": ["Nội dung B", "Nội dung A", "Nội dung C"],
    })
    encoder = FakeEncoder()

    result = create_embeddings(
        articles,
        encoder,
        model_name="fake-model",
        data_version="test-v1",
    )

    assert result["article_id"].tolist() == ["BAI_B", "BAI_A", "BAI_C"]
    assert len(result) == 3

    vectors = np.stack(result["embedding"].to_numpy())
    assert vectors.shape == (3, 2)
    assert np.isfinite(vectors).all()

    assert encoder.received_texts[0].startswith("Tin B")
    assert encoder.received_texts[1].startswith("Tin A")
    assert encoder.received_texts[2].startswith("Tin C")


def test_tu_choi_vector_co_nan():
    class BadEncoder:
        def encode(self, texts, **kwargs):
            return np.array([[np.nan, 1.0]], dtype=np.float32)

    articles = pd.DataFrame({
        "article_id": ["BAI_1"],
        "title": ["Tiêu đề"],
        "content": ["Nội dung"],
    })

    with pytest.raises(ValueError, match="NaN"):
        create_embeddings(articles, BadEncoder())


def test_tu_choi_article_id_trung():
    articles = pd.DataFrame({
        "article_id": ["BAI_1", "BAI_1"],
        "title": ["Tin một", "Tin hai"],
        "content": ["Nội dung một", "Nội dung hai"],
    })

    with pytest.raises(ValueError, match="trùng"):
        create_embeddings(articles, FakeEncoder())


def test_dung_cot_da_xu_ly_cua_p0():
    articles = pd.DataFrame({
        "article_id": ["BAI_1"],
        "title": ["Tiêu đề gốc"],
        "content": ["Nội dung gốc"],
        "processed_title": ["Tiêu đề P0"],
        "processed_content": ["Nội dung P0"],
    })
    encoder = FakeEncoder()

    create_embeddings(
        articles,
        encoder,
        title_col="processed_title",
        content_col="processed_content",
    )

    assert encoder.received_texts == ["Tiêu đề P0. Nội dung P0"]


def test_luu_va_tai_lai_artifact(tmp_path):
    articles = pd.DataFrame({
        "article_id": ["BAI_B", "BAI_A"],
        "title": ["Tin B", "Tin A"],
        "content": ["Nội dung B", "Nội dung A"],
    })
    original = create_embeddings(
        articles,
        FakeEncoder(),
        model_name="fake-model",
        data_version="test-v1",
    )

    metadata = {
        "embedding_run": "fake_title_content",
        "content_words": 60,
        "dataset": {"split": "train"},
    }

    path = tmp_path / "embeddings.npz"
    save_embedding_artifact(original, path, metadata=metadata)
    loaded = load_embedding_artifact(path)

    assert loaded["article_id"].tolist() == ["BAI_B", "BAI_A"]
    assert loaded["model_name"].tolist() == ["fake-model", "fake-model"]
    assert loaded["data_version"].tolist() == ["test-v1", "test-v1"]
    assert loaded.attrs["embedding_metadata"] == metadata

    np.testing.assert_allclose(
        np.stack(loaded["embedding"].to_numpy()),
        np.stack(original["embedding"].to_numpy()),
    )