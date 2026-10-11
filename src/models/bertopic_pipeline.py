from pathlib import Path
from typing import Any

import yaml
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP


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


def build_bertopic_model(
    config: dict[str, Any],
    embedding_model: Any | None = None,
) -> BERTopic:
    """Khởi tạo BERTopic theo cấu hình baseline của M13."""
    umap_config = config["umap"]
    hdbscan_config = config["hdbscan"]
    topic_config = config["topic_representation"]

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
        top_n_words=topic_config["top_n_words"],
        verbose=False,
    )