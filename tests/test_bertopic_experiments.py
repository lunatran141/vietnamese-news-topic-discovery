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
        "validation_adjusted_rand_index",
        "validation_normalized_mutual_info",
        "validation_npmi_coherence",
        "fit_seconds",
        "validation_predict_seconds",
        "runtime_seconds",
    }

    assert required_metrics.issubset(metrics)

def test_e2_changes_only_umap_neighbors() -> None:
    """E2 chỉ được đổi n_neighbors so với E1."""
    runs = {
        run["run_id"]: run
        for run in load_experiment_config()["runs"]
    }

    selected = runs["e1_vietnamese_sbert"]
    experiment = runs[
        "e2_vietnamese_sbert_neighbors30"
    ]

    changed_umap_fields = {
        key
        for key in selected["umap"]
        if selected["umap"][key] != experiment["umap"][key]
    }

    assert experiment["embedding"] == selected["embedding"]
    assert experiment["hdbscan"] == selected["hdbscan"]
    assert changed_umap_fields == {"n_neighbors"}
    assert experiment["umap"]["n_neighbors"] == 30


def test_e3_changes_only_hdbscan_cluster_size() -> None:
    """E3 chỉ được đổi min_cluster_size so với E1."""
    runs = {
        run["run_id"]: run
        for run in load_experiment_config()["runs"]
    }

    selected = runs["e1_vietnamese_sbert"]
    experiment = runs[
        "e3_vietnamese_sbert_cluster30"
    ]

    changed_hdbscan_fields = {
        key
        for key in selected["hdbscan"]
        if (
            selected["hdbscan"][key]
            != experiment["hdbscan"][key]
        )
    }

    assert experiment["embedding"] == selected["embedding"]
    assert experiment["umap"] == selected["umap"]
    assert changed_hdbscan_fields == {"min_cluster_size"}
    assert experiment["hdbscan"]["min_cluster_size"] == 30


def test_title_ablation_changes_only_input_text() -> None:
    """E4 phải giữ nguyên mô hình E1 và chỉ đổi đầu vào sang tiêu đề."""
    runs = {
        run["run_id"]: run
        for run in load_experiment_config()["runs"]
    }
    title_content = runs["e1_vietnamese_sbert"]
    title_only = runs["e4_vietnamese_sbert_title_only"]

    assert title_content["input_text_column"] == "text"
    assert title_content["input_text_label"] == "title_content"
    assert title_only["input_text_column"] == "processed_title"
    assert title_only["input_text_label"] == "title_only"
    assert title_only["embedding"] == title_content["embedding"]
    assert title_only["umap"] == title_content["umap"]
    assert title_only["hdbscan"] == title_content["hdbscan"]
