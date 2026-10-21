"""Kiểm thử đơn vị và tích hợp cho pipeline phân cụm K-Means (Task M02)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.kmeans import (
    compute_clustering_metrics,
    compute_purity,
    compute_stability,
    extract_cluster_keywords,
    find_representative_docs,
    load_kmeans_artifacts,
    predict_kmeans,
    save_kmeans_artifacts,
    train_kmeans,
)


def test_train_kmeans_basic() -> None:
    """Kiểm tra khởi tạo và huấn luyện mô hình K-Means cơ bản."""
    # Tạo ma trận thưa giả lập 10 bài báo, 4 đặc trưng
    X = csr_matrix([
        [1.0, 0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
        [0.8, 0.2, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.9, 0.1, 0.0],
        [0.0, 0.8, 0.2, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.9, 0.1],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.1, 0.9],
    ])

    model = train_kmeans(X, n_clusters=3, random_state=42)
    assert isinstance(model, KMeans)
    assert model.n_clusters == 3
    assert model.cluster_centers_.shape == (3, 4)

    # Ngoại lệ n_clusters không hợp lệ
    with pytest.raises(ValueError, match="n_clusters phải lớn hơn hoặc bằng 2"):
        train_kmeans(X, n_clusters=1)

    with pytest.raises(ValueError, match="không thể vượt quá số mẫu"):
        train_kmeans(X, n_clusters=15)

    with pytest.raises(ValueError, match="Ma trận huấn luyện rỗng"):
        train_kmeans(csr_matrix((0, 4)), n_clusters=2)


def test_predict_kmeans_and_distances() -> None:
    """Kiểm tra gán cụm và tính toán khoảng cách đến tâm."""
    X = csr_matrix([
        [1.0, 0.0],
        [1.0, 0.1],
        [0.0, 1.0],
        [0.1, 1.0],
    ])
    model = train_kmeans(X, n_clusters=2, random_state=42)
    labels, dists = predict_kmeans(model, X)

    assert len(labels) == 4
    assert len(dists) == 4
    assert set(labels).issubset({0, 1})
    assert (dists >= 0.0).all()
    assert np.isfinite(dists).all()


def test_compute_clustering_metrics_unsupervised() -> None:
    """Kiểm tra tính toán các chỉ số nội tại (Silhouette, DBI, CH, Inertia)."""
    X = csr_matrix([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.9, 0.1],
    ])
    model = train_kmeans(X, n_clusters=2, random_state=42)
    labels, _ = predict_kmeans(model, X)

    metrics = compute_clustering_metrics(X, labels, model=model)
    assert "inertia" in metrics
    assert "silhouette_score" in metrics
    assert "davies_bouldin_index" in metrics
    assert "calinski_harabasz_score" in metrics

    assert -1.0 <= metrics["silhouette_score"] <= 1.0
    assert metrics["davies_bouldin_index"] >= 0.0
    assert metrics["inertia"] >= 0.0


def test_compute_clustering_metrics_supervised() -> None:
    """Kiểm tra tính toán ARI, NMI, Purity trên nhãn đối sánh category_gold."""
    y_true = ["Thể thao", "Thể thao", "Kinh doanh", "Kinh doanh"]
    # Phân cụm hoàn hảo
    y_pred_perfect = [0, 0, 1, 1]

    purity = compute_purity(y_true, y_pred_perfect)
    assert purity == 1.0

    X = csr_matrix(np.eye(4))
    metrics = compute_clustering_metrics(X, y_pred_perfect, gold_labels=y_true)
    assert metrics["ari"] == 1.0
    assert metrics["nmi"] == 1.0
    assert metrics["purity"] == 1.0


def test_zero_label_leakage() -> None:
    """Xác nhận K-Means huấn luyện hoàn toàn không nhận category_gold."""
    import inspect
    sig = inspect.signature(train_kmeans)
    param_names = list(sig.parameters.keys())
    assert "gold_labels" not in param_names
    assert "category_gold" not in param_names
    assert "labels" not in param_names


def test_extract_cluster_keywords() -> None:
    """Kiểm tra trích xuất từ khóa trọng số lớn nhất từ tâm cụm."""
    X = csr_matrix([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
    ])
    feature_names = ["bóng_đá", "chứng_khoán", "thời_tiết"]
    model = train_kmeans(X, n_clusters=2, random_state=42)

    keywords = extract_cluster_keywords(model, feature_names, top_n=2)
    assert len(keywords) == 2
    assert "0" in keywords
    assert "1" in keywords

    # Mỗi cụm có đúng 2 từ khóa
    assert len(keywords["0"]) == 2
    assert len(keywords["1"]) == 2

    # Trọng số giảm dần
    assert keywords["0"][0]["weight"] >= keywords["0"][1]["weight"]


def test_find_representative_docs() -> None:
    """Kiểm tra chọn bài báo đại diện gần tâm nhất (khoảng cách nhỏ nhất lên đầu)."""
    cluster_ids = [0, 0, 1, 1]
    distances = [0.85, 0.12, 0.95, 0.20]
    article_ids = ["A1", "A2", "B1", "B2"]
    meta_df = pd.DataFrame({
        "article_id": ["A1", "A2", "B1", "B2"],
        "title": ["Bài A1 xa", "Bài A2 gần", "Bài B1 xa", "Bài B2 gần"],
        "category_gold": ["Thể thao", "Thể thao", "Kinh tế", "Kinh tế"],
    })

    reps = find_representative_docs(cluster_ids, distances, article_ids, meta_df, top_n=2)

    # Cụm 0: A2 phải đứng đầu vì khoảng cách 0.12 < 0.85
    assert reps["0"][0]["article_id"] == "A2"
    assert reps["0"][0]["rank"] == 1
    assert reps["0"][1]["article_id"] == "A1"
    assert reps["0"][1]["rank"] == 2

    # Cụm 1: B2 phải đứng đầu vì 0.20 < 0.95
    assert reps["1"][0]["article_id"] == "B2"
    assert reps["1"][0]["rank"] == 1


def test_save_and_load_kmeans_artifacts_roundtrip(tmp_path: Path) -> None:
    """Kiểm tra lưu và nạp lại trọn vẹn bộ artifacts"""
    X = csr_matrix([[1.0, 0.0], [0.0, 1.0], [1.0, 0.5], [0.5, 1.0]])
    model = train_kmeans(X, n_clusters=2, random_state=42)
    labels, dists = predict_kmeans(model, X)

    doc_topics_df = pd.DataFrame({
        "article_id": ["ID1", "ID2", "ID3", "ID4"],
        "title": ["T1", "T2", "T3", "T4"],
        "cluster_id": labels,
        "distance_to_center": dists,
        "split": ["train"] * 4,
        "category_gold": ["C1", "C2", "C1", "C2"],
    })

    config = {"n_clusters": 2, "random_state": 42}
    run_meta = {"duration": 1.0, "purity": 1.0}
    kws = {"0": [{"term": "từ1", "weight": 0.5}], "1": [{"term": "từ2", "weight": 0.6}]}
    reps = {"0": [{"rank": 1, "article_id": "ID1"}], "1": [{"rank": 1, "article_id": "ID2"}]}

    out_dir = tmp_path / "artifacts" / "kmeans"
    save_kmeans_artifacts(
        output_dir=out_dir,
        model=model,
        config=config,
        run_metadata=run_meta,
        document_topics_df=doc_topics_df,
        topic_keywords=kws,
        representative_docs=reps,
    )

    expected_files = [
        "kmeans_model.joblib",
        "document_topics.csv",
        "topic_keywords.json",
        "representative_docs.json",
        "config.json",
        "run_metadata.json",
    ]
    for fn in expected_files:
        assert (out_dir / fn).exists(), f"Thiếu file: {fn}"

    loaded = load_kmeans_artifacts(out_dir)
    assert loaded["config"] == config
    assert loaded["run_metadata"] == run_meta
    assert len(loaded["document_topics"]) == 4
    assert loaded["topic_keywords"] == kws
    assert loaded["representative_docs"] == reps

    new_labels, new_dists = predict_kmeans(loaded["model"], X)
    np.testing.assert_array_equal(new_labels, labels)
    np.testing.assert_allclose(new_dists, dists)


def test_official_kmeans_artifacts_exist_and_conform_to_handbook() -> None:
    """Kiểm tra toàn bộ bộ artifacts chính thức trong artifacts/kmeans/ và results/."""
    artifacts_dir = Path("artifacts/kmeans")
    if not artifacts_dir.exists():
        pytest.skip("Chưa tạo artifacts/kmeans")

    loaded = load_kmeans_artifacts(artifacts_dir)
    doc_topics = loaded["document_topics"]

    required_cols = {"article_id", "title", "cluster_id", "distance_to_center", "split", "category_gold"}
    assert required_cols.issubset(doc_topics.columns)

    assert len(doc_topics) == 9274
    assert doc_topics["split"].value_counts().to_dict() == {
        "train": 6493,
        "val": 1391,
        "test": 1390,
    }

    assert (doc_topics["distance_to_center"] >= 0.0).all()
    assert np.isfinite(doc_topics["distance_to_center"]).all()

    figures_dir = Path("results/kmeans/figures")
    for fig_name in [
        "kmeans_elbow_inertia.png",
        "kmeans_silhouette_davies.png",
        "kmeans_cluster_size_distribution.png",
        "kmeans_2d_clusters.png",
        "kmeans_silhouette_samples.png",
    ]:
        assert (figures_dir / fig_name).exists(), f"Thiếu biểu đồ: {fig_name}"

    k_metrics_path = Path("results/kmeans/kmeans_k_search_metrics.csv")
    assert k_metrics_path.exists()
    k_df = pd.read_csv(k_metrics_path)
    assert len(k_df) == 7  # 7 giá trị K khảo sát: [3, 5, 7, 9, 11, 14, 18]


def test_distance_matches_centroid_norm() -> None:
    """distance_to_center phải bằng khoảng cách Euclid tới tâm của cụm được gán."""
    X = csr_matrix([
        [1.0, 0.0, 0.2],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.1],
        [0.1, 0.9, 0.0],
        [0.0, 0.0, 1.0],
        [0.1, 0.0, 0.9],
    ])
    model = train_kmeans(X, n_clusters=3, random_state=42)
    labels, dists = predict_kmeans(model, X)

    dense = X.toarray()
    for index, (label, dist) in enumerate(zip(labels, dists)):
        expected = np.linalg.norm(dense[index] - model.cluster_centers_[label])
        assert dist == pytest.approx(expected, abs=1e-6)


def test_compute_stability() -> None:
    """Độ ổn định không phụ thuộc cách đánh số cụm và không nhận nhãn gold."""
    import inspect

    labels_a = np.array([0, 0, 1, 1, 2, 2])
    labels_b = np.array([2, 2, 0, 0, 1, 1])  # cùng phân hoạch, khác tên cụm
    mean_score, std_score = compute_stability([labels_a, labels_b])
    assert mean_score == pytest.approx(1.0)
    assert std_score == pytest.approx(0.0)

    labels_c = np.array([0, 1, 0, 1, 0, 1])
    lower_mean, _ = compute_stability([labels_a, labels_c])
    assert lower_mean < 1.0

    with pytest.raises(ValueError, match="ít nhất 2 bộ nhãn"):
        compute_stability([labels_a])

    parameter_names = set(inspect.signature(compute_stability).parameters)
    assert not {"gold_labels", "category_gold", "y_true"} & parameter_names


def test_official_k_search_table_has_single_best_and_selected() -> None:
    """Bảng K-search chính thức có đúng một K theo composite và đúng một K được chọn."""
    path = Path("results/kmeans/kmeans_k_search_metrics.csv")
    if not path.exists():
        pytest.skip("Chưa chạy K-search")

    table = pd.read_csv(path)
    required = {
        "k", "silhouette_score", "davies_bouldin_index", "nmi", "purity",
        "min_cluster_size_val", "max_cluster_size_val", "n_small_clusters_val",
        "stability_ari_mean", "stability_ari_std", "train_inertia_drop",
        "is_best", "is_selected",
    }
    assert required.issubset(table.columns)
    assert int(table["is_best"].sum()) == 1
    assert int(table["is_selected"].sum()) == 1

    # Kích thước cụm nhỏ nhất phải dương và không vượt quá lớn nhất
    assert (table["min_cluster_size_val"] > 0).all()
    assert (table["min_cluster_size_val"] <= table["max_cluster_size_val"]).all()

    # Mô hình đã lưu phải dùng đúng K được chọn
    selected_k = int(table.loc[table["is_selected"], "k"].iloc[0])
    artifacts = load_kmeans_artifacts(Path("artifacts/kmeans"))
    assert artifacts["model"].n_clusters == selected_k
    assert artifacts["config"]["clustering"]["n_clusters"] == selected_k


def test_official_manual_evaluation_schema() -> None:
    """Bảng đánh giá thủ công có đủ cột, đủ cụm cho mỗi K và trạng thái hợp lệ."""
    path = Path("results/kmeans/kmeans_manual_evaluation.csv")
    if not path.exists():
        pytest.skip("Chưa sinh bảng đánh giá thủ công")

    table = pd.read_csv(path)
    required = {
        "k", "cluster_id", "size_train", "size_val", "top_keywords",
        "nearest_titles_train", "random_titles_val", "dominant_gold_val",
        "manual_label", "coherence_1_5", "is_mixed_topic", "notes", "review_status",
    }
    assert required.issubset(table.columns)

    for k_value, group in table.groupby("k"):
        assert sorted(group["cluster_id"]) == list(range(int(k_value)))
        # Tổng số bài Train của các cụm phải bằng số bài Train
        assert int(group["size_train"].sum()) == 6493
        assert int(group["size_val"].sum()) == 1391

    allowed_status = {"not_reviewed", "draft_assistant_pending_confirmation", "confirmed"}
    assert set(table["review_status"]).issubset(allowed_status)

    # Cấu hình K=9 đã được phê duyệt chính thức
    k9_status = table.loc[table["k"] == 9, "review_status"]
    assert (k9_status == "confirmed").all()

    # Hàng đã có điểm thì điểm phải nằm trong 1..5
    scored = table["coherence_1_5"].dropna()
    assert scored.between(1, 5).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

