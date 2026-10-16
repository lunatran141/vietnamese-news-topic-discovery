"""Thực thi pipeline thực nghiệm bóc tách tiền xử lý P0 đến P3 (Task M03)"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

# Console UTF-8 trên Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

# Matplotlib headless
os.environ["MPLCONFIGDIR"] = str(Path(".matplotlib_cache").resolve())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.data.dataset_loader import load_split_dataset
from src.features.tfidf import (
    create_tfidf_vectorizer,
    evaluate_validation_vocabulary,
    load_stopwords,
    transform_tfidf_pipeline,
)
from src.models.kmeans import (
    compute_clustering_metrics,
    compute_stability,
    extract_cluster_keywords,
    find_representative_docs,
    predict_kmeans,
    train_kmeans,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chạy thực nghiệm bóc tách tiền xử lý P0 đến P3 (Task M03)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/m03_preprocessing_ablation.yaml",
        help="Đường dẫn file cấu hình YAML",
    )
    return parser.parse_args()


def compute_lexical_interpretability_metrics(
    cluster_keywords: dict[str, list[dict[str, Any]]],
    reference_stopwords: set[str],
) -> dict[str, float]:
    """Tính các chỉ số chất lượng từ vựng và khả năng diễn giải chủ đề."""
    all_top10_terms = []
    total_keywords = 0
    compound_count = 0
    stopword_pollution_count = 0

    for _, kw_list in cluster_keywords.items():
        top15 = kw_list[:15]
        top10 = kw_list[:10]
        all_top10_terms.extend([k["term"].lower() for k in top10])

        for kw in top15:
            term = kw["term"].lower()
            total_keywords += 1
            if "_" in term:
                compound_count += 1
            if term in reference_stopwords:
                stopword_pollution_count += 1

    # Topic Diversity: tỷ lệ từ khóa không trùng lặp trong top-10 giữa các cụm
    unique_top10 = len(set(all_top10_terms))
    total_top10 = len(all_top10_terms)
    topic_diversity = round(unique_top10 / total_top10, 4) if total_top10 > 0 else 0.0

    # Tỷ lệ từ ghép trong top-15
    compound_ratio = (
        round((compound_count / total_keywords) * 100.0, 2)
        if total_keywords > 0
        else 0.0
    )

    # Tỷ lệ nhiễu từ dừng trong top-15
    stopword_ratio = (
        round((stopword_pollution_count / total_keywords) * 100.0, 2)
        if total_keywords > 0
        else 0.0
    )

    return {
        "topic_diversity_top10": topic_diversity,
        "compound_keyword_ratio_pct": compound_ratio,
        "stopword_keyword_ratio_pct": stopword_ratio,
        "stopword_pollution_count_top15": stopword_pollution_count,
    }


def plot_m03_visualizations(
    metrics_k9_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Tạo bộ 4 biểu đồ trực quan hóa nghiên cứu chất lượng cao (300 DPI)."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )

    variants = metrics_k9_df["variant_id"].tolist()
    labels = [f"{vid}\n({vname})" for vid, vname in zip(variants, metrics_k9_df["variant_name"])]
    x = np.arange(len(variants))
    width = 0.22

    # Biểu đồ so sánh các chỉ số ngoại vi & độ ổn định (K=9)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    rects1 = ax.bar(x - 1.5 * width, metrics_k9_df["purity"] * 100, width, label="Purity (%)", color="#1f77b4")
    rects2 = ax.bar(x - 0.5 * width, metrics_k9_df["nmi"] * 100, width, label="NMI (x100)", color="#2ca02c")
    rects3 = ax.bar(x + 0.5 * width, metrics_k9_df["ari"] * 100, width, label="ARI (x100)", color="#ff7f0e")
    rects4 = ax.bar(x + 1.5 * width, metrics_k9_df["stability_ari_mean"] * 100, width, label="Stability ARI (x100)", color="#9467bd")

    ax.set_title("So sánh chỉ số đánh giá cụm ngoại vi và độ ổn định qua P0 -> P3 (K=9)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Điểm số (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", frameon=True)

    for rects in [rects1, rects2, rects3, rects4]:
        for bar in rects:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    fig.savefig(figures_dir / "m03_ablation_metrics_comparison.png", dpi=300)
    plt.close(fig)

    # Biểu đồ nghịch lý chỉ số nội tại vs ngoại vi
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax2 = ax1.twinx()

    color_pur = "#1f77b4"
    color_dbi = "#d62728"
    color_nmi = "#2ca02c"

    l1 = ax1.plot(x, metrics_k9_df["purity"] * 100, marker="o", color=color_pur, linewidth=2.5, label="Purity")
    l2 = ax1.plot(x, metrics_k9_df["nmi"] * 100, marker="s", color=color_nmi, linewidth=2.5, linestyle="--", label="NMI")
    ax1.set_ylabel("Chỉ số ngoại vi: Purity & NMI (%)", color=color_pur, fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_pur)
    ax1.set_ylim(50, 95)

    l3 = ax2.plot(x, metrics_k9_df["davies_bouldin_index"], marker="^", color=color_dbi, linewidth=2.5, linestyle="-.", label="Davies-Bouldin")
    ax2.set_ylabel("Davies-Bouldin Index", color=color_dbi, fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_dbi)
    ax2.set_ylim(5.0, 6.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_title("So sánh chỉ số nội tại với chất lượng chủ đề thực tế", fontsize=12, fontweight="bold", pad=12)

    lines = l1 + l2 + l3
    leg_labels = [l.get_label() for l in lines]
    ax1.legend(lines, leg_labels, loc="lower right", frameon=True)

    plt.tight_layout()
    fig.savefig(figures_dir / "m03_intrinsic_vs_extrinsic_paradox.png", dpi=300)
    plt.close(fig)

    # Biểu đồ chất lượng từ vựng & khả năng diễn giải
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bar_width = 0.35
    b1 = ax.bar(x - bar_width / 2, metrics_k9_df["compound_keyword_ratio_pct"], bar_width, label="Tỷ lệ từ ghép có nghĩa (%)", color="#2ca02c")
    b2 = ax.bar(x + bar_width / 2, metrics_k9_df["stopword_keyword_ratio_pct"], bar_width, label="Tỷ lệ nhiễu stopwords (%)", color="#d62728")

    ax.set_title("Chất lượng diễn giải từ khóa tâm cụm: Tỷ lệ từ ghép và nhiễu stopwords (K=9)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Tỷ lệ trong top-15 từ khóa (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", frameon=True)

    for bar in b1:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for bar in b2:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(figures_dir / "m03_lexical_interpretability.png", dpi=300)
    plt.close(fig)

    # 4. Biểu đồ kiểm tra độ nhạy trên cả K=7 (mốc chuẩn) và K=9 (cấu hình chọn)
    if not sensitivity_df.empty:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        k7_data = sensitivity_df[sensitivity_df["k"] == 7].set_index("variant_id")
        k9_data = sensitivity_df[sensitivity_df["k"] == 9].set_index("variant_id")

        v_ids = ["P0", "P1", "P2", "P3"]
        k7_purity = [k7_data.loc[v, "purity"] * 100 for v in v_ids]
        k9_purity = [k9_data.loc[v, "purity"] * 100 for v in v_ids]
        k7_nmi = [k7_data.loc[v, "nmi"] * 100 for v in v_ids]
        k9_nmi = [k9_data.loc[v, "nmi"] * 100 for v in v_ids]

        x_sens = np.arange(len(v_ids))
        w = 0.2

        r1 = ax.bar(x_sens - 1.5 * w, k7_purity, w, label="Purity (mốc chuẩn K=7)", color="#6baed6")
        r2 = ax.bar(x_sens - 0.5 * w, k9_purity, w, label="Purity (K=9 lựa chọn)", color="#08519c")
        r3 = ax.bar(x_sens + 0.5 * w, k7_nmi, w, label="NMI (mốc chuẩn K=7)", color="#74c476")
        r4 = ax.bar(x_sens + 1.5 * w, k9_nmi, w, label="NMI (K=9 lựa chọn)", color="#006d2c")

        ax.set_title("Kiểm tra tính nhất quán của xu hướng tiền xử lý trên cả K=7 và K=9", fontsize=12, fontweight="bold", pad=12)
        ax.set_ylabel("Điểm số (%)", fontsize=11)
        ax.set_xticks(x_sens)
        ax.set_xticklabels(v_ids, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.legend(loc="lower right", frameon=True)

        for rects in [r1, r2, r3, r4]:
            for bar in rects:
                h = bar.get_height()
                ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        fig.savefig(figures_dir / "m03_k7_vs_k9_sensitivity.png", dpi=300)
        plt.close(fig)

    print(f"Đã lưu toàn bộ 4 biểu đồ trực quan hóa nghiên cứu vào: {figures_dir}")


def main() -> None:
    start_total_time = time.time()
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy cấu hình: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    dataset_cfg = config["dataset"]
    vectorizer_cfg = config["vectorizer"]
    clustering_cfg = config["clustering"]
    output_cfg = config["output"]

    split_manifest_path = Path(dataset_cfg["split_manifest_path"])
    results_dir = Path(output_cfg.get("results_dir", "results/kmeans"))
    figures_dir = Path(output_cfg.get("figures_dir", "results/kmeans/figures"))
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Nạp danh sách từ dừng làm căn cứ đo lường độ nhiễu
    ref_sw_path = config.get("reference", {}).get("stopwords_file", "configs/vietnamese_stopwords.txt")
    reference_stopwords_list = load_stopwords(ref_sw_path) or []
    reference_stopwords_set = set(reference_stopwords_list)
    print(f"Nạp {len(reference_stopwords_set)} từ dừng tham chiếu để đo tỷ lệ nhiễu.")

    variants = dataset_cfg["variants"]
    primary_k = clustering_cfg.get("primary_k", 9)
    sensitivity_k_list = clustering_cfg.get("sensitivity_k_list", [7, 9])
    stability_seeds = clustering_cfg.get("stability_seeds", [42, 43, 44, 45, 46])
    random_state = clustering_cfg.get("random_state", 42)

    primary_records: list[dict[str, Any]] = []
    sensitivity_records: list[dict[str, Any]] = []
    all_cluster_keywords_dump: dict[str, Any] = {}
    qualitative_rows: list[dict[str, Any]] = []

    print("\n" + "=" * 70)
    print("BẮT ĐẦU THỰC NGHIỆM BÓC TÁCH TIỀN XỬ LÝ P0 -> P3 (TASK M03)")
    print("=" * 70)

    for var in variants:
        var_id = var["id"]
        var_name = var["name"]
        var_path = Path(var["path"])
        var_desc = var["description"]

        print(f"\n>>> Đang xử lý biến thể: [{var_id}] {var_name}")
        print(f"    Mô tả: {var_desc}")
        t0 = time.time()

        # Nạp dữ liệu train và validation
        train_df = load_split_dataset(var_path, split_manifest_path, split_name="train")
        val_df = load_split_dataset(var_path, split_manifest_path, split_name="val")

        train_texts = train_df["text"].tolist()
        val_texts = val_df["text"].tolist()
        val_gold = val_df["category_gold"].tolist()

        # Khởi tạo và huấn luyện TfidfVectorizer 
        vec_config = {
            "ngram_range": vectorizer_cfg.get("ngram_range", [1, 1]),
            "min_df": vectorizer_cfg.get("min_df", 2),
            "max_df": vectorizer_cfg.get("max_df", 0.85),
            "max_features": vectorizer_cfg.get("max_features", 20000),
            "sublinear_tf": vectorizer_cfg.get("sublinear_tf", True),
            "norm": vectorizer_cfg.get("norm", "l2"),
            "token_pattern": vectorizer_cfg.get("token_pattern", r"(?u)\b\w\w+\b"),
            "lowercase": vectorizer_cfg.get("lowercase", False),
            "stopwords_file": vectorizer_cfg.get("stopwords_file", None),
        }

        vectorizer = create_tfidf_vectorizer(vec_config)
        X_train = vectorizer.fit_transform(train_texts)
        X_val = transform_tfidf_pipeline(vectorizer, val_texts)
        feature_names = list(vectorizer.get_feature_names_out())
        vocab_size = len(feature_names)

        # Đo lường không gian đặc trưng trên tập Validation
        val_vocab_eval = evaluate_validation_vocabulary(vectorizer, val_texts)

        print(f"    Train shape: {X_train.shape} | Val shape: {X_val.shape} | Vocab: {vocab_size}")
        print(f"    Val Sparsity: {val_vocab_eval['val_sparsity_pct']}% | OOV Rate: {val_vocab_eval['val_oov_rate_pct']}% | Mean active terms: {val_vocab_eval['val_mean_active_terms']}")

        # Chạy phân cụm K-Means cho từng K trong sensitivity_k_list
        for k in sensitivity_k_list:
            km_model = train_kmeans(
                X_train,
                n_clusters=k,
                init=clustering_cfg.get("init", "k-means++"),
                n_init=clustering_cfg.get("n_init", 10),
                max_iter=clustering_cfg.get("max_iter", 300),
                random_state=random_state,
            )

            train_clusters, train_dists = predict_kmeans(km_model, X_train)
            val_clusters, val_dists = predict_kmeans(km_model, X_val)

            # Chỉ số phân cụm trên tập Validation
            val_metrics = compute_clustering_metrics(
                X_val,
                val_clusters,
                model=km_model,
                gold_labels=val_gold,
            )

            val_sizes = np.bincount(val_clusters, minlength=k)
            min_size_val = int(val_sizes.min())
            max_size_val = int(val_sizes.max())
            n_small_val = int((val_sizes < 30).sum())

            # Đo độ ổn định đa seed (5 seeds)
            stability_mean, stability_std = float("nan"), float("nan")
            other_seeds = [s for s in stability_seeds if s != random_state]
            if other_seeds:
                seed_labels = [val_clusters]
                for seed in other_seeds:
                    s_model = train_kmeans(
                        X_train,
                        n_clusters=k,
                        init=clustering_cfg.get("init", "k-means++"),
                        n_init=clustering_cfg.get("n_init", 10),
                        max_iter=clustering_cfg.get("max_iter", 300),
                        random_state=seed,
                    )
                    s_pred, _ = predict_kmeans(s_model, X_val)
                    seed_labels.append(s_pred)
                stability_mean, stability_std = compute_stability(seed_labels)

            # Trích xuất từ khóa tâm cụm
            cluster_kws = extract_cluster_keywords(km_model, feature_names, top_n=15)
            all_cluster_keywords_dump[f"{var_id}_k{k}"] = cluster_kws

            # Đo các chỉ số chất lượng diễn giải từ vựng
            lex_metrics = compute_lexical_interpretability_metrics(
                cluster_kws, reference_stopwords_set
            )

            elapsed = round(time.time() - t0, 2)

            sens_record = {
                "variant_id": var_id,
                "variant_name": var_name,
                "k": k,
                "vocab_size": vocab_size,
                "val_sparsity_pct": val_vocab_eval["val_sparsity_pct"],
                "val_oov_rate_pct": val_vocab_eval["val_oov_rate_pct"],
                "val_mean_active_terms": val_vocab_eval["val_mean_active_terms"],
                "val_zero_vector_docs": val_vocab_eval["val_zero_vector_docs"],
                "train_inertia": round(km_model.inertia_, 2),
                "val_inertia": round(float(np.sum(val_dists**2)), 2),
                "silhouette_score": val_metrics.get("silhouette_score", 0.0),
                "davies_bouldin_index": val_metrics.get("davies_bouldin_index", 0.0),
                "calinski_harabasz_score": val_metrics.get("calinski_harabasz_score", 0.0),
                "ari": val_metrics.get("ari", 0.0),
                "nmi": val_metrics.get("nmi", 0.0),
                "purity": val_metrics.get("purity", 0.0),
                "stability_ari_mean": round(stability_mean, 4),
                "stability_ari_std": round(stability_std, 4),
                "min_cluster_size_val": min_size_val,
                "max_cluster_size_val": max_size_val,
                "n_small_clusters_val": n_small_val,
                "topic_diversity_top10": lex_metrics["topic_diversity_top10"],
                "compound_keyword_ratio_pct": lex_metrics["compound_keyword_ratio_pct"],
                "stopword_keyword_ratio_pct": lex_metrics["stopword_keyword_ratio_pct"],
                "stopword_pollution_count_top15": lex_metrics["stopword_pollution_count_top15"],
                "runtime_seconds": elapsed,
            }
            sensitivity_records.append(sens_record)

            if k == primary_k:
                primary_records.append(sens_record)
                print(
                    f"    [K={k}] Silhouette: {sens_record['silhouette_score']:+.4f} | "
                    f"DBI: {sens_record['davies_bouldin_index']:.4f} | "
                    f"NMI: {sens_record['nmi']:.4f} | "
                    f"Purity: {sens_record['purity']:.4f} | "
                    f"ARI: {sens_record['ari']:.4f} | "
                    f"Stability: {sens_record['stability_ari_mean']:.4f}"
                )
                print(
                    f"           Từ ghép: {sens_record['compound_keyword_ratio_pct']}% | "
                    f"Nhiễu từ dừng: {sens_record['stopword_keyword_ratio_pct']}% | "
                    f"Đa dạng từ khóa: {sens_record['topic_diversity_top10']}"
                )

                # Trích xuất dữ liệu định tính chi tiết cho từng cụm ở K=9
                rep_docs = find_representative_docs(
                    train_clusters,
                    train_dists,
                    train_df["article_id"].astype(str).tolist(),
                    train_df[["article_id", "title", "category_gold", "source", "published_at"]],
                    top_n=1,
                )

                gold_series = pd.Series(val_gold)
                for cid in range(k):
                    c_mask = (val_clusters == cid)
                    c_gold_sub = gold_series[c_mask]
                    if len(c_gold_sub) > 0:
                        gold_val_counts = c_gold_sub.value_counts()
                        dominant_gold = gold_val_counts.index[0]
                        dominant_share = round(float(gold_val_counts.iloc[0] / len(c_gold_sub)) * 100.0, 1)
                    else:
                        dominant_gold = "N/A"
                        dominant_share = 0.0

                    top10_kws = [kw["term"] for kw in cluster_kws[str(cid)][:10]]
                    top10_str = ", ".join(top10_kws)

                    # Đếm số từ dừng và số từ ghép trong cụm này
                    c_sw_count = sum(1 for kw in top10_kws if kw.lower() in reference_stopwords_set)
                    c_compound_count = sum(1 for kw in top10_kws if "_" in kw)

                    nearest_title = rep_docs.get(str(cid), [{}])[0].get("title", "")

                    qualitative_rows.append({
                        "variant_id": var_id,
                        "variant_name": var_name,
                        "cluster_id": cid,
                        "size_val": int(c_mask.sum()),
                        "dominant_gold_category": dominant_gold,
                        "dominant_gold_share_pct": dominant_share,
                        "compound_words_top10": c_compound_count,
                        "stopwords_polluted_top10": c_sw_count,
                        "top10_keywords": top10_str,
                        "nearest_article_title": nearest_title,
                    })

    # Lưu trữ bảng kết quả
    primary_df = pd.DataFrame(primary_records)
    sensitivity_df = pd.DataFrame(sensitivity_records)
    qualitative_df = pd.DataFrame(qualitative_rows)

    metrics_csv_path = results_dir / "m03_preprocessing_ablation_metrics.csv"
    sens_csv_path = results_dir / "m03_preprocessing_ablation_k7_sensitivity.csv"
    qual_csv_path = results_dir / "m03_qualitative_cluster_examples.csv"
    kws_json_path = results_dir / "m03_cluster_keywords_p0_p3.json"

    primary_df.to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")
    sensitivity_df.to_csv(sens_csv_path, index=False, encoding="utf-8-sig")
    qualitative_df.to_csv(qual_csv_path, index=False, encoding="utf-8-sig")

    with open(kws_json_path, "w", encoding="utf-8") as f:
        json.dump(all_cluster_keywords_dump, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"Đã lưu bảng kết quả chính M03 (K=9) vào: {metrics_csv_path}")
    print(f"Đã lưu bảng độ nhạy K=7 vào: {sens_csv_path}")
    print(f"Đã lưu bảng đối chiếu định tính từng cụm vào: {qual_csv_path}")
    print(f"Đã lưu từ khóa chi tiết vào: {kws_json_path}")

    # Vẽ 4 biểu đồ trực quan hóa nghiên cứu
    plot_m03_visualizations(primary_df, sensitivity_df, figures_dir)

    total_time = round(time.time() - start_total_time, 2)
    print(f"\nHoàn tất toàn bộ thực nghiệm Task M03 thành công sau {total_time} giây!")


if __name__ == "__main__":
    main()

