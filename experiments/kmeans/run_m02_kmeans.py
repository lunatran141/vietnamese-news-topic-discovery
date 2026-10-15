"""Thực thi pipeline phân cụm K-Means Baseline cho Task M02."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

# Reconfigure console UTF-8 on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

# Set Matplotlib headless config
os.environ["MPLCONFIGDIR"] = str(Path(".matplotlib_cache").resolve())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_samples, silhouette_score
import yaml

from src.data.dataset_loader import load_split_dataset
from src.features.tfidf import load_tfidf_artifacts
from src.models.kmeans import (
    compute_clustering_metrics,
    compute_stability,
    extract_cluster_keywords,
    find_representative_docs,
    predict_kmeans,
    save_kmeans_artifacts,
    train_kmeans,
)


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Chạy K-Means Clustering Pipeline cho Task M02"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kmeans.yaml",
        help="Đường dẫn file cấu hình YAML (mặc định: configs/kmeans.yaml)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Chỉ định trực tiếp số cụm K để huấn luyện (bỏ qua auto-select)",
    )
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Bỏ qua bước K-search, dùng default_k trong YAML",
    )
    return parser.parse_args()


def run_k_search(
    train_matrix: Any,
    val_matrix: Any,
    val_gold: list[str],
    k_range: list[int],
    clustering_cfg: dict[str, Any],
    results_dir: Path,
    selected_k: int | None = None,
) -> tuple[int, pd.DataFrame]:
    """Khảo sát số cụm K trên tập Validation và xuất bảng số liệu."""
    
    print(f"\nBắt đầu K-search trên tập Validation với các giá trị K = {k_range}...")
    records: list[dict[str, Any]] = []

    random_state = clustering_cfg.get("random_state", 42)
    n_init = clustering_cfg.get("n_init", 10)
    max_iter = clustering_cfg.get("max_iter", 300)
    init_method = clustering_cfg.get("init", "k-means++")
    stability_seeds = [
        seed for seed in clustering_cfg.get("stability_seeds", []) if seed != random_state
    ]
    small_cluster_threshold = clustering_cfg.get("small_cluster_threshold", 30)

    for k in k_range:
        t0 = time.time()
        # Huấn luyện mô hình K-Means trên train_matrix
        model = train_kmeans(
            train_matrix,
            n_clusters=k,
            init=init_method,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
        )

        # Dự đoán và tính khoảng cách trên val_matrix
        val_clusters, val_dists = predict_kmeans(model, val_matrix)

        # Đo lường chỉ số
        metrics = compute_clustering_metrics(
            val_matrix,
            val_clusters,
            model=model,
            gold_labels=val_gold,
        )
        runtime = round(time.time() - t0, 2)

        val_inertia = float(np.sum(val_dists ** 2))
        val_sizes = np.bincount(val_clusters, minlength=k)

        # Độ ổn định qua nhiều seed
        stability_mean, stability_std = float("nan"), float("nan")
        if stability_seeds:
            label_sets = [val_clusters]
            for seed in stability_seeds:
                seed_model = train_kmeans(
                    train_matrix,
                    n_clusters=k,
                    init=init_method,
                    n_init=n_init,
                    max_iter=max_iter,
                    random_state=seed,
                )
                label_sets.append(predict_kmeans(seed_model, val_matrix)[0])
            stability_mean, stability_std = compute_stability(label_sets)

        record = {
            "k": k,
            "train_inertia": round(metrics.get("inertia", 0.0), 2),
            "val_inertia": round(val_inertia, 2),
            "silhouette_score": metrics.get("silhouette_score", 0.0),
            "davies_bouldin_index": metrics.get("davies_bouldin_index", 0.0),
            "calinski_harabasz_score": metrics.get("calinski_harabasz_score", 0.0),
            "ari": metrics.get("ari", 0.0),
            "nmi": metrics.get("nmi", 0.0),
            "purity": metrics.get("purity", 0.0),
            "min_cluster_size_val": int(val_sizes.min()),
            "max_cluster_size_val": int(val_sizes.max()),
            "n_small_clusters_val": int((val_sizes < small_cluster_threshold).sum()),
            "stability_ari_mean": round(stability_mean, 4),
            "stability_ari_std": round(stability_std, 4),
            "runtime_seconds": runtime,
        }
        records.append(record)

        print(
            f"  - K={k:2d}: Silhouette={record['silhouette_score']:+.4f} | "
            f"DBI={record['davies_bouldin_index']:.4f} | "
            f"NMI={record['nmi']:.4f} | "
            f"Purity={record['purity']:.4f} | "
            f"Stability={record['stability_ari_mean']:.4f} | "
            f"MinSize={record['min_cluster_size_val']} | "
            f"Time={runtime}s"
        )

    metrics_df = pd.DataFrame(records)

    # Mức giảm Train Inertia so với K trước đó, dùng để kiểm tra có điểm khuỷu hay không
    metrics_df["train_inertia_drop"] = (-metrics_df["train_inertia"].diff()).round(2)

    sil_norm = (metrics_df["silhouette_score"] - metrics_df["silhouette_score"].min()) / (
        metrics_df["silhouette_score"].max() - metrics_df["silhouette_score"].min() + 1e-8
    )
    dbi_norm = (metrics_df["davies_bouldin_index"].max() - metrics_df["davies_bouldin_index"]) / (
        metrics_df["davies_bouldin_index"].max() - metrics_df["davies_bouldin_index"].min() + 1e-8
    )
    composite_score = sil_norm * 0.5 + dbi_norm * 0.5
    metrics_df["composite_score"] = composite_score.round(4)

    best_idx = metrics_df["composite_score"].idxmax()
    metrics_df["is_best"] = False
    metrics_df.loc[best_idx, "is_best"] = True
    composite_k = int(metrics_df.loc[best_idx, "k"])

    final_k = composite_k if selected_k is None else selected_k
    if final_k not in set(metrics_df["k"]):
        raise ValueError(f"K đã chọn ({final_k}) không nằm trong dải khảo sát {k_range}")
    metrics_df["is_selected"] = metrics_df["k"] == final_k

    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "kmeans_k_search_metrics.csv"
    metrics_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nĐã lưu kết quả K-search vào: {csv_path}")
    print(f"K theo composite_score (tham khảo): {composite_k}")
    print(f"K được chọn cuối cùng: {final_k}")

    return final_k, metrics_df


def generate_visualizations(
    metrics_df: pd.DataFrame,
    final_model: Any,
    train_matrix: Any,
    train_clusters: np.ndarray,
    figures_dir: Path,
    val_matrix: Any | None = None,
    val_clusters: np.ndarray | None = None,
) -> None:
    """Vẽ và lưu 5 biểu đồ trực quan hóa nghiên cứu."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Biểu đồ Elbow (Inertia vs K)
    if "train_inertia" in metrics_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(metrics_df["k"], metrics_df["train_inertia"], marker="o", color="#1f77b4", linewidth=2, label="Train Inertia")
        ax.plot(metrics_df["k"], metrics_df["val_inertia"], marker="s", color="#ff7f0e", linewidth=2, linestyle="--", label="Val Inertia")
        ax.set_title("K-Means Elbow Method (Inertia vs K)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Số cụm K", fontsize=11)
        ax.set_ylabel("Inertia", fontsize=11)
        ax.set_xticks(metrics_df["k"])
        ax.legend()
        plt.tight_layout()
        fig.savefig(figures_dir / "kmeans_elbow_inertia.png", dpi=300)
        plt.close(fig)

    # 2. Biểu đồ Silhouette & Davies-Bouldin
    if "silhouette_score" in metrics_df.columns:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        color1 = "#2ca02c"
        color2 = "#d62728"

        ax1.set_xlabel("Số cụm K", fontsize=11)
        ax1.set_ylabel("Silhouette Score", color=color1, fontsize=11)
        line1 = ax1.plot(metrics_df["k"], metrics_df["silhouette_score"], color=color1, marker="o", linewidth=2, label="Silhouette Score")
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_xticks(metrics_df["k"])

        ax2 = ax1.twinx()
        ax2.set_ylabel("Davies-Bouldin Index", color=color2, fontsize=11)
        line2 = ax2.plot(metrics_df["k"], metrics_df["davies_bouldin_index"], color=color2, marker="^", linewidth=2, linestyle="--", label="Davies-Bouldin")
        ax2.tick_params(axis="y", labelcolor=color2)

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="best")
        plt.title("Chỉ số Đánh giá Cụm Validation theo K", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        fig.savefig(figures_dir / "kmeans_silhouette_davies.png", dpi=300)
        plt.close(fig)

    # 3. Phân bố số lượng bài báo trong từng cụm
    cluster_counts = pd.Series(train_clusters).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        cluster_counts.index.astype(str),
        cluster_counts.values,
        color="#3b528b",
        edgecolor="black",
        alpha=0.85,
    )
    ax.set_title(f"Phân bố Số lượng Bài báo giữa các Cụm (K={len(cluster_counts)})", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Mã Cụm (Cluster ID)", fontsize=11)
    ax.set_ylabel("Số lượng Bài báo", fontsize=11)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 15, f"{int(yval)}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    fig.savefig(figures_dir / "kmeans_cluster_size_distribution.png", dpi=300)
    plt.close(fig)

    # 4. Trực quan hóa 2D các cụm (TruncatedSVD chiếu 20,000 chiều -> 2D)
    print("Đang tính toán chiếu không gian 2D bằng TruncatedSVD...")
    svd = TruncatedSVD(n_components=2, random_state=42)
    # Lấy mẫu tối đa 2,000 bài để vẽ biểu đồ phân tán rõ nét, không bị chồng đè
    sample_size = min(2000, train_matrix.shape[0])
    sample_indices = np.random.RandomState(42).choice(train_matrix.shape[0], size=sample_size, replace=False)
    sample_matrix = train_matrix[sample_indices]
    sample_labels = train_clusters[sample_indices]

    coords_2d = svd.fit_transform(sample_matrix)

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.get_cmap("tab10" if len(cluster_counts) <= 10 else "tab20")
    scatter = ax.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=sample_labels,
        cmap=cmap,
        alpha=0.6,
        s=20,
        edgecolors="none",
    )
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(len(cluster_counts)))
    cbar.set_label("Cluster ID", fontsize=11)
    ax.set_title(f"Không gian Phân cụm K-Means 2D (TruncatedSVD, N={sample_size})", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Thành phần SVD 1", fontsize=11)
    ax.set_ylabel("Thành phần SVD 2", fontsize=11)
    plt.tight_layout()
    fig.savefig(figures_dir / "kmeans_2d_clusters.png", dpi=300)
    plt.close(fig)

    # 5. Biểu đồ Silhouette Analysis từng mẫu (theo từng cụm trên tập Validation)
    if val_matrix is not None and val_clusters is not None:
        n_clusters = len(np.unique(val_clusters))
        sample_sil_values = silhouette_samples(val_matrix, val_clusters, metric="euclidean")
        sil_avg = float(silhouette_score(val_matrix, val_clusters, metric="euclidean"))

        fig, ax = plt.subplots(figsize=(9, 6))
        y_lower = 10
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

        for i in range(n_clusters):
            ith_cluster_sil = sample_sil_values[val_clusters == i]
            ith_cluster_sil.sort()
            size_cluster_i = ith_cluster_sil.shape[0]
            y_upper = y_lower + size_cluster_i

            color = colors[i]
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_sil,
                facecolor=color,
                edgecolor=color,
                alpha=0.75,
                label=f"Cụm {i} (n={size_cluster_i})",
            )
            ax.text(-0.028, y_lower + 0.5 * size_cluster_i, f"Cụm {i}", fontsize=9, va="center", fontweight="bold")
            y_lower = y_upper + 10

        ax.set_title(f"Silhouette Plot for K-Means (K={n_clusters})", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Hệ số Silhouette", fontsize=11)
        ax.set_ylabel("Chỉ mục bài báo trong từng cụm", fontsize=11)
        ax.axvline(x=sil_avg, color="red", linestyle="--", linewidth=1.5, label=f"Trung bình ({sil_avg:.4f})")
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
        ax.legend(loc="lower right", frameon=True, fontsize=9)

        plt.tight_layout()
        fig.savefig(figures_dir / "kmeans_silhouette_samples.png", dpi=300)
        plt.close(fig)

    print(f"Đã lưu toàn bộ 5 biểu đồ trực quan vào: {figures_dir}")


def main() -> None:
    """Hàm chính điều phối pipeline M02."""
    start_time = time.time()
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy cấu hình: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    dataset_cfg = config["dataset"]
    clustering_cfg = config["clustering"]
    topics_cfg = config.get("topics", {})
    output_dir = Path(config.get("output", {}).get("artifacts_dir", "artifacts/kmeans"))
    results_dir = Path(config.get("output", {}).get("results_dir", "results/kmeans"))
    figures_dir = results_dir / "figures"

    # Nạp ma trận TF-IDF
    tfidf_dir = Path(dataset_cfg.get("tfidf_artifacts_dir", "artifacts/tfidf"))
    print(f"Đang nạp ma trận TF-IDF từ: {tfidf_dir}")
    train_bundle = load_tfidf_artifacts(tfidf_dir, split="train")
    val_bundle = load_tfidf_artifacts(tfidf_dir, split="val")

    train_matrix = train_bundle["matrix"]
    train_ids = train_bundle["article_ids"]
    val_matrix = val_bundle["matrix"]
    val_ids = val_bundle["article_ids"]
    feature_names = train_bundle["feature_names"]

    print(f"Train Shape: {train_matrix.shape}, Val Shape: {val_matrix.shape}")

    # Đọc metadata bài báo
    processed_path = Path(dataset_cfg.get("processed_dataset_path", "data/processed/p2_segmented.csv"))
    print(f"Đang nạp metadata bài viết từ: {processed_path}")
    raw_df = pd.read_csv(processed_path)
    metadata_lookup = {
        str(row["article_id"]): {
            "title": str(row.get("title", "")),
            "category_gold": str(row.get("category_gold", "")),
            "source": str(row.get("source", "")),
            "published_at": str(row.get("published_at", "")),
        }
        for _, row in raw_df.iterrows()
    }

    val_gold = [metadata_lookup.get(aid, {}).get("category_gold", "") for aid in val_ids]
    train_gold = [metadata_lookup.get(aid, {}).get("category_gold", "") for aid in train_ids]

    # K-search luôn chạy để bảng số liệu/biểu đồ đầy đủ
    k_range = clustering_cfg.get("k_search_range", [3, 5, 7, 9, 11, 14, 18])
    chosen_k = args.k if args.k is not None else clustering_cfg.get("selected_k")
    if args.skip_search:
        best_k = chosen_k if chosen_k is not None else clustering_cfg.get("default_k", 7)
        print(f"Bỏ qua K-search, sử dụng K = {best_k}")
        metrics_df = pd.DataFrame()
    else:
        best_k, metrics_df = run_k_search(
            train_matrix,
            val_matrix,
            val_gold,
            k_range,
            clustering_cfg,
            results_dir,
            selected_k=chosen_k,
        )

    # Huấn luyện mô hình tối ưu (FIT DUY NHẤT TRÊN TRAIN)
    print(f"\nHuấn luyện mô hình K-Means tối ưu với K = {best_k} trên tập Train (Zero Leakage)...")
    final_model = train_kmeans(
        train_matrix,
        n_clusters=best_k,
        init=clustering_cfg.get("init", "k-means++"),
        n_init=clustering_cfg.get("n_init", 10),
        max_iter=clustering_cfg.get("max_iter", 300),
        random_state=clustering_cfg.get("random_state", 42),
    )

    # Dự đoán nhãn và khoảng cách đến tâm cho Train và Val
    train_clusters, train_dists = predict_kmeans(final_model, train_matrix)
    val_clusters, val_dists = predict_kmeans(final_model, val_matrix)

    # Đo lường chỉ số chính thức của mô hình tối ưu trên Val
    val_eval = compute_clustering_metrics(
        val_matrix,
        val_clusters,
        model=final_model,
        gold_labels=val_gold,
    )
    print("\nChỉ số kiểm định chính thức trên tập Validation:")
    print(f"  - Inertia:                {val_eval.get('inertia', 0.0):.2f}")
    print(f"  - Silhouette Score:       {val_eval.get('silhouette_score', 0.0):+.4f}")
    print(f"  - Davies-Bouldin Index:   {val_eval.get('davies_bouldin_index', 0.0):.4f}")
    print(f"  - Calinski-Harabasz:      {val_eval.get('calinski_harabasz_score', 0.0):.2f}")
    print(f"  - Adjusted Rand Index:    {val_eval.get('ari', 0.0):.4f}")
    print(f"  - Normalized Mutual Info: {val_eval.get('nmi', 0.0):.4f}")
    print(f"  - Purity:                 {val_eval.get('purity', 0.0):.4f}")

    # Trích xuất từ khóa tâm cụm (Centroid Keywords)
    top_n_keywords = topics_cfg.get("top_n_keywords", 15)
    cluster_keywords = extract_cluster_keywords(final_model, feature_names, top_n=top_n_keywords)

    # Trích xuất bài báo đại diện gần tâm nhất (Representative Docs)
    top_n_reps = topics_cfg.get("representative_docs_per_cluster", 5)
    representative_docs = find_representative_docs(
        train_clusters,
        train_dists,
        train_ids,
        raw_df,
        top_n=top_n_reps,
    )

    # Bảng phân cụm document_topics.csv (Train + Val)
    train_records = []
    for aid, cid, dist, cat in zip(train_ids, train_clusters, train_dists, train_gold):
        meta = metadata_lookup.get(aid, {})
        train_records.append({
            "article_id": aid,
            "title": meta.get("title", ""),
            "cluster_id": int(cid),
            "distance_to_center": round(float(dist), 4),
            "split": "train",
            "category_gold": cat,
        })

    val_records = []
    for aid, cid, dist, cat in zip(val_ids, val_clusters, val_dists, val_gold):
        meta = metadata_lookup.get(aid, {})
        val_records.append({
            "article_id": aid,
            "title": meta.get("title", ""),
            "cluster_id": int(cid),
            "distance_to_center": round(float(dist), 4),
            "split": "val",
            "category_gold": cat,
        })

    document_topics_df = pd.DataFrame(train_records + val_records)

    # Tạo run_metadata.json
    duration = round(time.time() - start_time, 2)
    cluster_counts = pd.Series(train_clusters).value_counts().sort_index().to_dict()

    run_metadata = {
        "pipeline_name": "M02_kmeans_baseline",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_duration_seconds": duration,
        "n_clusters": best_k,
        "random_state": clustering_cfg.get("random_state", 42),
        "cluster_counts_train": {str(k): int(v) for k, v in cluster_counts.items()},
        "metrics_validation": val_eval,
        "k_search_summary": {
            "status": "completed" if not metrics_df.empty else "skipped",
            "k_range": k_range if not metrics_df.empty else [best_k],
        },
    }

    final_config_dict = {
        "dataset": dataset_cfg,
        "clustering": dict(clustering_cfg, n_clusters=best_k),
        "topics": topics_cfg,
        "selected_at": datetime.now(timezone.utc).isoformat(),
    }

    # Lưu trữ toàn bộ 6 artifacts
    print(f"\nLưu trữ artifacts vào: {output_dir}")
    save_kmeans_artifacts(
        output_dir=output_dir,
        model=final_model,
        config=final_config_dict,
        run_metadata=run_metadata,
        document_topics_df=document_topics_df,
        topic_keywords=cluster_keywords,
        representative_docs=representative_docs,
    )

    # Xuất hình ảnh nghiên cứu
    print("\nĐang tạo các biểu đồ trực quan hóa nghiên cứu...")
    generate_visualizations(
        metrics_df if not metrics_df.empty else pd.DataFrame([{"k": best_k, "train_inertia": final_model.inertia_, "val_inertia": float(np.sum(val_dists**2)), "silhouette_score": val_eval["silhouette_score"], "davies_bouldin_index": val_eval["davies_bouldin_index"]}]),
        final_model,
        train_matrix,
        train_clusters,
        figures_dir,
        val_matrix=val_matrix,
        val_clusters=val_clusters,
    )

    print("\nHoàn tất pipeline Task M02 K-Means thành công!")
    print(f"Tổng thời gian: {duration} giây")
    print("\nTop 5 từ khóa tiêu biểu của từng cụm:")
    for cid, kws in cluster_keywords.items():
        terms_str = ", ".join([f"{kw['term']} ({kw['weight']:.2f})" for kw in kws[:5]])
        print(f"  - Cụm {cid} (Size: {cluster_counts.get(int(cid), 0)} bài): {terms_str}")


if __name__ == "__main__":
    main()

