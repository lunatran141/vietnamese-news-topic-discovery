from pathlib import Path

from src.models.bertopic_pipeline import (
    build_bertopic_model,
    load_bertopic_config,
)


CONFIG_PATH = Path("configs/bertopic.yaml")


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

    assert model.top_n_words == 10