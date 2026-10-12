from pathlib import Path
from typing import Any

from src.models.bertopic_pipeline import (
    build_bertopic_model,
    build_ctfidf_model,
    build_hdbscan_model,
    build_umap_model,
    build_vectorizer_model,
    encode_documents,
    fit_bertopic_model,
    load_bertopic_config,
    load_stopwords,
)
CONFIG_PATH = Path("configs/bertopic.yaml")

class FakeEmbeddingModel:
    """Mô phỏng embedding model để test không phải tải model thật."""

    def __init__(self) -> None:
        self.encode_arguments: dict[str, Any] = {}

    def encode(self, documents: list[str], **kwargs: Any) -> list[list[float]]:
        self.encode_arguments = kwargs
        return [[float(index), 1.0] for index, _ in enumerate(documents)]


class FakeTopicModel:
    """Mô phỏng BERTopic để kiểm tra dữ liệu truyền vào khi fit."""

    def __init__(self) -> None:
        self.documents: list[str] = []
        self.embeddings: list[list[float]] = []

    def fit_transform(
        self,
        documents: list[str],
        embeddings: list[list[float]],
    ) -> tuple[list[int], list[float]]:
        self.documents = documents
        self.embeddings = embeddings
        return [0, 1], [0.8, 0.9]

def test_load_bertopic_config() -> None:
    """Kiểm tra các cấu hình chính của BERTopic baseline."""
    config = load_bertopic_config(CONFIG_PATH)

    assert config["dataset"]["fit_split"] == "train"
    assert config["dataset"]["validation_split"] == "val"
    assert config["dataset"]["test_split"] == "test"
    assert config["dataset"]["text_variant"] == "text"

    assert config["embedding"]["model_name"] == (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    assert config["experiment"]["seed"] == 42


def test_build_bertopic_model_from_config() -> None:
    """Kiểm tra UMAP và HDBSCAN nhận đúng tham số baseline."""
    config = load_bertopic_config(CONFIG_PATH)
    model = build_bertopic_model(
        config,
        embedding_model=None,
    )

    assert model.umap_model.n_neighbors == 15
    assert model.umap_model.n_components == 5
    assert model.umap_model.metric == "cosine"

    assert model.hdbscan_model.min_cluster_size == 15
    assert model.hdbscan_model.min_samples == 5
    assert model.hdbscan_model.prediction_data is True

    assert model.vectorizer_model.ngram_range == (1, 2)
    assert "và" in model.vectorizer_model.stop_words
    assert model.ctfidf_model is not None

    assert model.top_n_words == 10

def test_encode_documents_uses_embedding_config() -> None:
    """Kiểm tra tham số encode được lấy đúng từ cấu hình."""
    config = load_bertopic_config(CONFIG_PATH)
    embedding_model = FakeEmbeddingModel()
    documents = ["bài viết kinh tế", "bài viết thể thao"]

    embeddings = encode_documents(
        embedding_model,
        documents,
        config,
    )

    assert len(embeddings) == 2
    assert embedding_model.encode_arguments["batch_size"] == 32
    assert embedding_model.encode_arguments["normalize_embeddings"] is True


def test_fit_bertopic_model_uses_documents_and_embeddings() -> None:
    """Kiểm tra BERTopic được fit bằng văn bản và embedding tương ứng."""
    topic_model = FakeTopicModel()
    documents = ["bài viết kinh tế", "bài viết thể thao"]
    embeddings = [[0.0, 1.0], [1.0, 1.0]]

    topics, probabilities = fit_bertopic_model(
        topic_model,
        documents,
        embeddings,
    )

    assert topics == [0, 1]
    assert probabilities == [0.8, 0.9]
    assert topic_model.documents == documents
    assert topic_model.embeddings == embeddings

def test_load_vietnamese_stopwords() -> None:
    """Kiểm tra hư từ bị loại nhưng từ phủ định vẫn được giữ."""
    stopwords = load_stopwords(
        "configs/vietnamese_stopwords.txt"
    )

    assert "và" in stopwords
    assert "của" in stopwords
    assert "không" not in stopwords

def test_build_pipeline_components() -> None:
    """Kiểm tra từng thành phần nhận đúng cấu hình."""
    config = load_bertopic_config(CONFIG_PATH)

    vectorizer = build_vectorizer_model(config)
    ctfidf = build_ctfidf_model(config)
    umap_model = build_umap_model(config)
    hdbscan_model = build_hdbscan_model(config)

    assert vectorizer.ngram_range == (1, 2)
    assert "và" in vectorizer.stop_words

    assert ctfidf is not None

    assert umap_model.n_neighbors == 15
    assert umap_model.n_components == 5
    assert umap_model.metric == "cosine"

    assert hdbscan_model.min_cluster_size == 15
    assert hdbscan_model.min_samples == 5
    assert hdbscan_model.prediction_data is True
