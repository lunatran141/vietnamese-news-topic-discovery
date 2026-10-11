from pathlib import Path
from typing import Any

import yaml
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer

def load_bertopic_config(config_path: str | Path) -> dict[str, Any]:
    """Đọc cấu hình BERTopic từ file YAML."""
    path = Path(config_path)

    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Cấu hình BERTopic phải có dạng dictionary.")

    return config

def load_stopwords(stopwords_path: str | Path) -> list[str]:
    """Đọc stopword và bỏ qua dòng trống hoặc dòng chú thích."""
    path = Path(stopwords_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy file stopword: {path}"
        )

    with path.open("r", encoding="utf-8-sig") as file:
        return [
            line.strip()
            for line in file
            if line.strip() and not line.lstrip().startswith("#")
        ]

def build_bertopic_model(
    config: dict[str, Any],
    embedding_model: Any | None = None,
) -> BERTopic:
    """Khởi tạo BERTopic theo cấu hình baseline của M13."""
    umap_config = config["umap"]
    hdbscan_config = config["hdbscan"]
    topic_config = config["topic_representation"]
    vectorizer_config = config["vectorizer"]
    stopwords = load_stopwords(
        vectorizer_config["stopwords_file"]
    )

    # Vectorizer loại hư từ và cho phép keyword gồm một hoặc hai từ.
    vectorizer_model = CountVectorizer(
        stop_words=stopwords,
        ngram_range=tuple(vectorizer_config["ngram_range"]),
        min_df=vectorizer_config["min_df"],
        max_df=vectorizer_config["max_df"],
    )

    # UMAP giảm chiều embedding trước khi phân cụm.
    umap_model = UMAP(
        n_neighbors=umap_config["n_neighbors"],
        n_components=umap_config["n_components"],
        min_dist=umap_config["min_dist"],
        metric=umap_config["metric"],
        random_state=umap_config["random_state"],
    )

    # HDBSCAN tự xác định số topic và cho phép xuất hiện bài nhiễu.
    hdbscan_model = HDBSCAN(
        min_cluster_size=hdbscan_config["min_cluster_size"],
        min_samples=hdbscan_config["min_samples"],
        metric=hdbscan_config["metric"],
        cluster_selection_method=hdbscan_config[
            "cluster_selection_method"
        ],
        prediction_data=hdbscan_config["prediction_data"],
    )

    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        top_n_words=topic_config["top_n_words"],
        verbose=False,
    )

def encode_documents(
    embedding_model: Any,
    documents: list[str],
    config: dict[str, Any],
) -> Any:
    """Tạo embedding cho danh sách văn bản theo cấu hình M13."""
    if not documents:
        raise ValueError("Danh sách văn bản không được rỗng.")

    embedding_config = config["embedding"]

    return embedding_model.encode(
        documents,
        batch_size=embedding_config["batch_size"],
        normalize_embeddings=embedding_config["normalize_embeddings"],
        show_progress_bar=True,
    )


def fit_bertopic_model(
    topic_model: Any,
    documents: list[str],
    embeddings: Any,
) -> tuple[Any, Any]:
    """Fit BERTopic bằng văn bản và embedding tương ứng."""
    if not documents:
        raise ValueError("Danh sách văn bản không được rỗng.")

    if len(documents) != len(embeddings):
        raise ValueError(
            "Số văn bản phải bằng số embedding."
        )

    topics, probabilities = topic_model.fit_transform(
        documents,
        embeddings,
    )

    return topics, probabilities