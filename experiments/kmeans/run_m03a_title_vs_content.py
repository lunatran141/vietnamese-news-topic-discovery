"""Thực nghiệm bóc tách phạm vi đầu vào: Chỉ tiêu đề vs Tiêu đề cộng nội dung (Task M03A).

Khảo sát tác động của độ dài văn bản đầu vào đối với mô hình K-Means trên nền dữ liệu
tiền xử lý P3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

# Đảm bảo thư mục gốc nằm trong sys.path khi chạy trực tiếp
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import yaml

from experiments.kmeans.run_m03_preprocessing_ablation import (
    compute_lexical_interpretability_metrics,
)
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
        description="Chạy thực nghiệm bóc tách đầu vào tiêu đề vs toàn bài (Task M03A)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/m03a_title_vs_content.yaml",
        help="Đường dẫn file cấu hình YAML",
    )
    return parser.parse_args()


def compute_category_breakdown(
    val_gold: list[str],
    val_clusters: np.ndarray,
    variant_id: str,
    k: int,
) -> list[dict[str, Any]]:
    """Đo lường độ tập trung của từng chuyên mục thực tế vào các cụm tương ứng."""
    gold_arr = np.array(val_gold)
    categories = sorted(list(set(val_gold)))
    breakdown_rows = []

    for cat in categories:
        cat_mask = gold_arr == cat
        cat_total = int(cat_mask.sum())
        if cat_total == 0:
            continue

        assigned_clusters = val_clusters[cat_mask]
        cluster_counts = pd.Series(assigned_clusters).value_counts()
        best_cluster = int(cluster_counts.index[0])
        best_cluster_docs = int(cluster_counts.iloc[0])

        cluster_mask = val_clusters == best_cluster
        cluster_total = int(cluster_mask.sum())
        cluster_purity = (
            round((best_cluster_docs / cluster_total) * 100.0, 2)
            if cluster_total > 0
            else 0.0
        )
        recall_pct = round((best_cluster_docs / cat_total) * 100.0, 2)

        breakdown_rows.append(
            {
                "variant_id": variant_id,
                "k": k,
                "category_gold": cat,
                "category_total_docs": cat_total,
                "dominant_cluster_id": best_cluster,
                "category_docs_in_dominant_cluster": best_cluster_docs,
                "cluster_total_docs": cluster_total,
                "category_recall_pct": recall_pct,
                "cluster_purity_pct": cluster_purity,
            }
        )

    return breakdown_rows


def main() -> None:
    start_total_time = time.time()
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    dataset_cfg = config["dataset"]
    vectorizer_cfg = config["vectorizer"]
    clustering_cfg = config["clustering"]
    output_cfg = config["output"]

    split_manifest_path = Path(dataset_cfg["split_manifest_path"])
    dataset_path = Path(dataset_cfg["processed_dataset_path"])
    results_dir = Path(output_cfg.get("results_dir", "results/kmeans"))
    results_dir.mkdir(parents=True, exist_ok=True)

    ref_sw_path = config.get("reference", {}).get(
        "stopwords_file", "configs/vietnamese_stopwords.txt"
    )
    reference_stopwords_list = load_stopwords(ref_sw_path) or []
    reference_stopwords_set = set(reference_stopwords_list)

    input_variants = dataset_cfg["input_variants"]
    primary_k = clustering_cfg.get("primary_k", 9)
    sensitivity_k_list = clustering_cfg.get("sensitivity_k_list", [7, 9])
    stability_seeds = clustering_cfg.get("stability_seeds", [42, 43, 44, 45, 46])
    random_state = clustering_cfg.get("random_state", 42)

    primary_records: list[dict[str, Any]] = []
    sensitivity_records: list[dict[str, Any]] = []
    category_breakdown_records: list[dict[str, Any]] = []
    all_cluster_keywords_dump: dict[str, Any] = {}
    qualitative_rows: list[dict[str, Any]] = []

    print("\n" + "=" * 70)
    print("BẮT ĐẦU THỰC NGHIỆM BÓC TÁCH ĐẦU VÀO: CHỈ TIÊU ĐỀ VS TOÀN BÀI (TASK M03A)")
    print(f"Tập dữ liệu nền: {dataset_path}")
    print("=" * 70)

    # Nạp sẵn tập dữ liệu theo split
    train_df = load_split_dataset(dataset_path, split_manifest_path, split_name="train")
    val_df = load_split_dataset(dataset_path, split_manifest_path, split_name="val")
    val_gold = val_df["category_gold"].tolist()

    for var in input_variants:
        var_id = var["id"]
        var_name = var["name"]
        text_col = var["text_column"]
        var_desc = var["description"]

        print(f"\n>>> Đang xử lý biến thể đầu vào: [{var_id}] {var_name}")
        print(f"    Cột dữ liệu: {text_col}")
        print(f"    Mô tả: {var_desc}")
        t_var_start = time.time()

        train_texts = train_df[text_col].fillna("").astype(str).tolist()
        val_texts = val_df[text_col].fillna("").astype(str).tolist()

        # Khởi tạo và huấn luyện TfidfVectorizer trên train
        t_vec_start = time.time()
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
        vectorize_time = round(time.time() - t_vec_start, 3)

        val_vocab_eval = evaluate_validation_vocabulary(vectorizer, val_texts)

        print(f"    Train shape: {X_train.shape} | Val shape: {X_val.shape} | Vocab: {vocab_size}")
        print(
            f"    Độ thưa Val: {val_vocab_eval['val_sparsity_pct']}% | "
            f"Tỷ lệ OOV: {val_vocab_eval['val_oov_rate_pct']}% | "
            f"Số bài vector 0: {val_vocab_eval['val_zero_vector_docs']}"
        )

        for k in sensitivity_k_list:
            t_k_start = time.time()
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

            # Đo độ ổn định qua 5 seeds
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

            cluster_kws = extract_cluster_keywords(km_model, feature_names, top_n=15)
            all_cluster_keywords_dump[f"{var_id}_k{k}"] = cluster_kws

            lex_metrics = compute_lexical_interpretability_metrics(
                cluster_kws, reference_stopwords_set
            )

            k_runtime = round(time.time() - t_k_start, 2)
            total_variant_runtime = round(time.time() - t_var_start, 2)

            sens_record = {
                "variant_id": var_id,
                "variant_name": var_name,
                "text_column": text_col,
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
                "vectorize_time_seconds": vectorize_time,
                "kmeans_runtime_seconds": k_runtime,
                "total_runtime_seconds": total_variant_runtime,
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

                # Phân tích theo từng chuyên mục cho K=9
                breakdown = compute_category_breakdown(val_gold, val_clusters, var_id, k)
                category_breakdown_records.extend(breakdown)

                # Dẫn chứng định tính cho K=9
                rep_docs = find_representative_docs(
                    train_clusters,
                    train_dists,
                    train_df["article_id"].astype(str).tolist(),
                    train_df[["article_id", "title", "category_gold", "source", "published_at"]],
                    top_n=1,
                )

                gold_series = pd.Series(val_gold)
                for cid in range(k):
                    c_mask = val_clusters == cid
                    c_gold_sub = gold_series[c_mask]
                    if len(c_gold_sub) > 0:
                        gold_val_counts = c_gold_sub.value_counts()
                        dominant_gold = gold_val_counts.index[0]
                        dominant_share = round(
                            float(gold_val_counts.iloc[0] / len(c_gold_sub)) * 100.0, 1
                        )
                    else:
                        dominant_gold = "N/A"
                        dominant_share = 0.0

                    top10_kws = [kw["term"] for kw in cluster_kws[str(cid)][:10]]
                    top10_str = ", ".join(top10_kws)
                    c_sw_count = sum(1 for kw in top10_kws if kw.lower() in reference_stopwords_set)
                    c_compound_count = sum(1 for kw in top10_kws if "_" in kw)

                    nearest_title = rep_docs.get(str(cid), [{}])[0].get("title", "")

                    qualitative_rows.append(
                        {
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
                        }
                    )

    # Lưu trữ các file kết quả
    primary_df = pd.DataFrame(primary_records)
    sensitivity_df = pd.DataFrame(sensitivity_records)
    breakdown_df = pd.DataFrame(category_breakdown_records)
    qualitative_df = pd.DataFrame(qualitative_rows)

    metrics_csv_path = results_dir / "m03a_title_vs_content_metrics.csv"
    sens_csv_path = results_dir / "m03a_title_vs_content_k7_sensitivity.csv"
    breakdown_csv_path = results_dir / "m03a_category_breakdown.csv"
    qual_csv_path = results_dir / "m03a_qualitative_cluster_examples.csv"
    kws_json_path = results_dir / "m03a_cluster_keywords.json"

    primary_df.to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")
    sensitivity_df.to_csv(sens_csv_path, index=False, encoding="utf-8-sig")
    breakdown_df.to_csv(breakdown_csv_path, index=False, encoding="utf-8-sig")
    qualitative_df.to_csv(qual_csv_path, index=False, encoding="utf-8-sig")

    with open(kws_json_path, "w", encoding="utf-8") as f:
        json.dump(all_cluster_keywords_dump, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"Đã lưu bảng kết quả chính M03A (K=9) vào: {metrics_csv_path}")
    print(f"Đã lưu bảng độ nhạy K=7 vào: {sens_csv_path}")
    print(f"Đã lưu bảng phân tích theo chuyên mục vào: {breakdown_csv_path}")
    print(f"Đã lưu bảng đối chiếu định tính từng cụm vào: {qual_csv_path}")
    print(f"Đã lưu từ khóa chi tiết vào: {kws_json_path}")

    total_time = round(time.time() - start_total_time, 2)
    print(f"\nHoàn tất toàn bộ thực nghiệm Task M03A thành công sau {total_time} giây!")


if __name__ == "__main__":
    main()
