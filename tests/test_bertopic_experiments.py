from pathlib import Path

import yaml


CONFIG_PATH = Path("configs/bertopic_experiments.yaml")


def load_experiment_config() -> dict:
    """Đọc cấu hình thí nghiệm M14."""
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_experiment_run_ids_are_unique() -> None:
    """Mỗi lần chạy phải có run_id riêng."""
    config = load_experiment_config()
    run_ids = [run["run_id"] for run in config["runs"]]

    assert len(run_ids) == len(set(run_ids))


def test_experiment_uses_train_and_validation_only() -> None:
    """M14 chỉ fit trên train và đánh giá trên validation."""
    experiment = load_experiment_config()["experiment"]

    assert experiment["fit_split"] == "train"
    assert experiment["evaluation_split"] == "val"
    assert experiment["test_split_locked"] is True
    assert experiment["fit_split"] != "test"
    assert experiment["evaluation_split"] != "test"


def test_embedding_runs_keep_clustering_config_fixed() -> None:
    """E0 và E1 chỉ được khác nhau ở embedding model."""
    runs = load_experiment_config()["runs"]
    baseline = runs[0]
    vietnamese_sbert = runs[1]

    assert baseline["embedding"]["model_name"] != (
        vietnamese_sbert["embedding"]["model_name"]
    )
    assert baseline["umap"] == vietnamese_sbert["umap"]
    assert baseline["hdbscan"] == vietnamese_sbert["hdbscan"]


def test_required_metrics_are_declared() -> None:
    """Cấu hình phải khai báo đủ metric dùng để so sánh."""
    metrics = set(
        load_experiment_config()["evaluation"]["metrics"]
    )

    required_metrics = {
        "topic_count",
        "train_outlier_rate",
        "validation_outlier_rate",
        "topic_diversity",
        "runtime_seconds",
    }

    assert required_metrics.issubset(metrics)