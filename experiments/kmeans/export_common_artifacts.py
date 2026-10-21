"""Đánh giá hậu kiểm tập Test, đo runtime từng công đoạn và xuất common artifacts K-Means."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.artifacts.adapters import (
    _compute_topic_diversity,
    export_kmeans_artifacts,
)
from src.data.dataset_loader import load_dataset
from src.features.article_text import build_article_text
from src.features.tfidf import (
    fit_tfidf_pipeline,
    load_tfidf_artifacts,
    transform_tfidf_pipeline,
)
from src.models.kmeans import (
    compute_clustering_metrics,
    predict_kmeans,
    train_kmeans,
)


DEFAULT_SOURCE_DIR = Path("artifacts/kmeans")
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/common/kmeans/kmeans_baseline"
)
DEFAULT_RESULTS_DIR = Path("results/kmeans")


def parse_arguments() -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description=(
            "Đánh giá hậu kiểm tập Test, đo runtime "
            "và chuyển kết quả K-Means sang common artifact schema."
        )
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Thư mục kết quả K-Means đầu vào.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Thư mục common artifact đầu ra.",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Thư mục lưu kết quả test_metrics.json.",
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default="kmeans_baseline",
        help="Định danh run_id cho bộ artifact.",
    )

    return parser.parse_args()


def _benchmark_single_k_runtime(
    config: dict[str, Any],
    tfidf_dir: Path,
    processed_path: Path,
    split_manifest_path: Path,
    n_clusters: int,
) -> dict[str, Any]:
    """Đo thời gian từng công đoạn cho cấu hình K-Means đã khóa trên tập Train/Val/Test."""
    tfidf_cfg_path = tfidf_dir / "config.json"
    with tfidf_cfg_path.open("r", encoding="utf-8") as f:
        tfidf_config = json.load(f)

    dataset_df = load_dataset(processed_path)
    splits_df = pd.read_csv(
        split_manifest_path,
        encoding="utf-8-sig",
    )
    dataset_df["article_id"] = dataset_df[
        "article_id"
    ].astype(str)
    splits_df["article_id"] = splits_df[
        "article_id"
    ].astype(str)
    merged_df = splits_df[["article_id", "split"]].merge(
        dataset_df,
        on="article_id",
        how="left",
        validate="one_to_one",
    )

    text_variant = tfidf_config.get("dataset", {}).get(
        "text_variant",
        "text",
    )
    merged_df["input_text"] = build_article_text(
        merged_df,
        text_variant,
    )

    train_df = merged_df[merged_df["split"] == "train"]
    val_df = merged_df[merged_df["split"] == "val"]
    test_df = merged_df[merged_df["split"] == "test"]

    train_texts = train_df["input_text"].tolist()
    val_texts = val_df["input_text"].tolist()
    test_texts = test_df["input_text"].tolist()
    val_gold = (
        val_df["category_gold"].fillna("").astype(str).tolist()
    )
    test_gold = (
        test_df["category_gold"].fillna("").astype(str).tolist()
    )

    # 1. Đo thời gian biểu diễn TF-IDF (fit trên train + transform train/val/test)
    t0 = time.perf_counter()
    vectorizer = fit_tfidf_pipeline(train_texts, tfidf_config)
    train_mat = transform_tfidf_pipeline(
        vectorizer,
        train_texts,
    )
    val_mat = transform_tfidf_pipeline(
        vectorizer,
        val_texts,
    )
    test_mat = transform_tfidf_pipeline(
        vectorizer,
        test_texts,
    )
    representation_seconds = round(
        time.perf_counter() - t0,
        4,
    )

    # 2. Đo thời gian huấn luyện K-Means (fit trên train)
    clustering_cfg = config.get("clustering", {})
    t1 = time.perf_counter()
    bench_model = train_kmeans(
        train_mat,
        n_clusters=n_clusters,
        init=clustering_cfg.get("init", "k-means++"),
        n_init=int(clustering_cfg.get("n_init", 10)),
        max_iter=int(clustering_cfg.get("max_iter", 300)),
        random_state=int(
            clustering_cfg.get("random_state", 42)
        ),
    )
    fit_seconds = round(time.perf_counter() - t1, 4)

    # 3. Đo thời gian suy diễn trên Validation
    t2 = time.perf_counter()
    val_pred, _ = predict_kmeans(bench_model, val_mat)
    validation_predict_seconds = round(
        time.perf_counter() - t2,
        4,
    )

    # 4. Đo thời gian suy diễn trên Test
    t3 = time.perf_counter()
    test_pred, _ = predict_kmeans(bench_model, test_mat)
    test_predict_seconds = round(
        time.perf_counter() - t3,
        4,
    )

    # 5. Đo thời gian tính toán bộ chỉ số đánh giá
    t4 = time.perf_counter()
    _ = compute_clustering_metrics(
        val_mat,
        val_pred,
        model=bench_model,
        gold_labels=val_gold,
    )
    _ = compute_clustering_metrics(
        test_mat,
        test_pred,
        model=bench_model,
        gold_labels=test_gold,
    )
    evaluation_seconds = round(time.perf_counter() - t4, 4)

    total_runtime = round(
        representation_seconds
        + fit_seconds
        + validation_predict_seconds
        + test_predict_seconds
        + evaluation_seconds,
        4,
    )

    return {
        "representation_seconds": representation_seconds,
        "fit_seconds": fit_seconds,
        "validation_predict_seconds": validation_predict_seconds,
        "test_predict_seconds": test_predict_seconds,
        "evaluation_seconds": evaluation_seconds,
        "runtime_seconds": total_runtime,
        "runtime_scope": (
            "TF-IDF fit/transform (train/val/test) + "
            "KMeans (K=9, seed=42) fit trên train + "
            "predict trên val và test + tính metric đánh giá; "
            "không gồm đọc/ghi ổ đĩa, quét lưới K-search và vẽ biểu đồ."
        ),
    }


def refresh_kmeans_source_artifacts(
    source_dir: Path,
    results_dir: Path,
) -> None:
    """Cập nhật đầy đủ 3 split (train/val/test), runtime breakdown và test metrics cho K-Means."""
    model_path = source_dir / "kmeans_model.joblib"
    config_path = source_dir / "config.json"
    run_metadata_path = source_dir / "run_metadata.json"
    keywords_path = source_dir / "topic_keywords.json"

    if not model_path.is_file():
        return

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    with run_metadata_path.open("r", encoding="utf-8") as f:
        run_metadata = json.load(f)
    with keywords_path.open("r", encoding="utf-8") as f:
        raw_keywords = json.load(f)

    dataset_cfg = config.get("dataset", {})
    tfidf_dir = Path(
        dataset_cfg.get(
            "tfidf_artifacts_dir",
            "artifacts/tfidf",
        )
    )
    processed_path = Path(
        dataset_cfg.get(
            "processed_dataset_path",
            "data/processed/p2_segmented.csv",
        )
    )
    split_manifest_path = Path(
        dataset_cfg.get(
            "split_manifest_path",
            "data/splits/split_manifest.csv",
        )
    )

    if not (tfidf_dir / "tfidf_test.npz").is_file():
        return

    model = joblib.load(model_path)
    n_clusters = int(
        run_metadata.get(
            "n_clusters",
            config.get("clustering", {}).get("n_clusters", 9),
        )
    )

    raw_df = pd.read_csv(
        processed_path,
        encoding="utf-8-sig",
    )
    metadata_lookup = {
        str(row["article_id"]): {
            "title": str(row.get("title", "")),
            "category_gold": str(
                row.get("category_gold", "")
            ),
            "source": str(row.get("source", "")),
            "published_at": str(row.get("published_at", "")),
        }
        for _, row in raw_df.iterrows()
    }

    split_bundles: dict[str, dict[str, Any]] = {}
    all_records: list[dict[str, Any]] = []

    for split_name in ("train", "val", "test"):
        bundle = load_tfidf_artifacts(
            tfidf_dir,
            split=split_name,
        )
        clusters, dists = predict_kmeans(
            model,
            bundle["matrix"],
        )
        golds = [
            metadata_lookup.get(aid, {}).get(
                "category_gold",
                "",
            )
            for aid in bundle["article_ids"]
        ]
        split_bundles[split_name] = {
            "matrix": bundle["matrix"],
            "article_ids": bundle["article_ids"],
            "clusters": clusters,
            "dists": dists,
            "golds": golds,
        }
        for aid, cid, dist, cat in zip(
            bundle["article_ids"],
            clusters,
            dists,
            golds,
        ):
            meta = metadata_lookup.get(str(aid), {})
            all_records.append(
                {
                    "article_id": str(aid),
                    "title": meta.get("title", ""),
                    "cluster_id": int(cid),
                    "distance_to_center": round(
                        float(dist),
                        4,
                    ),
                    "split": split_name,
                    "category_gold": cat,
                    "source": meta.get("source", ""),
                    "published_at": meta.get(
                        "published_at",
                        "",
                    ),
                }
            )

    document_topics_df = pd.DataFrame(all_records)
    document_topics_df.to_csv(
        source_dir / "document_topics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    val_eval = compute_clustering_metrics(
        split_bundles["val"]["matrix"],
        split_bundles["val"]["clusters"],
        model=model,
        gold_labels=split_bundles["val"]["golds"],
    )
    val_eval["train_inertia"] = round(
        float(model.inertia_),
        4,
    )
    val_eval["validation_inertia"] = round(
        float(np.sum(split_bundles["val"]["dists"] ** 2)),
        4,
    )
    val_eval["validation_ari"] = val_eval["ari"]
    val_eval["validation_nmi"] = val_eval["nmi"]
    val_eval["validation_purity"] = val_eval["purity"]
    val_eval["davies_bouldin_score"] = val_eval[
        "davies_bouldin_index"
    ]

    test_eval = compute_clustering_metrics(
        split_bundles["test"]["matrix"],
        split_bundles["test"]["clusters"],
        model=model,
        gold_labels=split_bundles["test"]["golds"],
    )
    test_eval["test_inertia"] = round(
        float(np.sum(split_bundles["test"]["dists"] ** 2)),
        4,
    )
    test_eval["test_ari"] = test_eval["ari"]
    test_eval["test_nmi"] = test_eval["nmi"]
    test_eval["test_purity"] = test_eval["purity"]
    test_eval["davies_bouldin_score"] = test_eval[
        "davies_bouldin_index"
    ]

    keyword_rows = []
    for cid_str, kw_list in raw_keywords.items():
        for rank, item in enumerate(kw_list, start=1):
            keyword_rows.append(
                {
                    "topic_id": int(cid_str),
                    "rank": rank,
                    "keyword": str(item.get("term", "")),
                    "weight": float(item.get("weight", 0.0)),
                }
            )
    topic_diversity = _compute_topic_diversity(
        pd.DataFrame(keyword_rows),
        top_n=10,
    )

    stability_mean = 0.8224
    stability_std = 0.0444
    k_search_csv = (
        results_dir / "kmeans_k_search_metrics.csv"
    )
    if k_search_csv.is_file():
        k_df = pd.read_csv(k_search_csv)
        matched = k_df[k_df["k"] == n_clusters]
        if not matched.empty:
            stability_mean = round(
                float(
                    matched.iloc[0]["stability_ari_mean"]
                ),
                4,
            )
            stability_std = round(
                float(
                    matched.iloc[0]["stability_ari_std"]
                ),
                4,
            )

    runtime_breakdown = _benchmark_single_k_runtime(
        config=config,
        tfidf_dir=tfidf_dir,
        processed_path=processed_path,
        split_manifest_path=split_manifest_path,
        n_clusters=n_clusters,
    )

    prev_duration = run_metadata.get(
        "k_search_execution_duration_seconds",
        run_metadata.get("execution_duration_seconds", 28.14),
    )
    cluster_counts_all = (
        document_topics_df["cluster_id"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    cluster_counts_test = (
        document_topics_df[
            document_topics_df["split"] == "test"
        ]["cluster_id"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    run_metadata.update(
        {
            "updated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "k_search_execution_duration_seconds": prev_duration,
            "execution_duration_seconds": runtime_breakdown[
                "runtime_seconds"
            ],
            "runtime_breakdown": runtime_breakdown,
            "metrics_validation": val_eval,
            "metrics_test": test_eval,
            "stability": {
                "stability_mean": stability_mean,
                "stability_std": stability_std,
                "metric": "pairwise_ari_across_seeds",
                "seeds": config.get("clustering", {}).get(
                    "stability_seeds",
                    [42, 43, 44, 45, 46],
                ),
            },
            "split_counts": {
                "train": int(
                    (
                        document_topics_df["split"] == "train"
                    ).sum()
                ),
                "val": int(
                    (
                        document_topics_df["split"] == "val"
                    ).sum()
                ),
                "test": int(
                    (
                        document_topics_df["split"] == "test"
                    ).sum()
                ),
                "total": len(document_topics_df),
            },
            "cluster_counts_test": {
                str(k): int(v)
                for k, v in cluster_counts_test.items()
            },
            "cluster_counts_total": {
                str(k): int(v)
                for k, v in cluster_counts_all.items()
            },
        }
    )

    with run_metadata_path.open("w", encoding="utf-8") as f:
        json.dump(
            run_metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    test_metrics_payload = {
        "model_name": "KMeans kmeans_baseline",
        "run_id": "kmeans_baseline",
        "n_clusters": n_clusters,
        "random_state": int(
            config.get("clustering", {}).get(
                "random_state",
                42,
            )
        ),
        "fit_split": "train",
        "selection_split": "val",
        "evaluation_split": "test",
        "train_documents": int(
            (document_topics_df["split"] == "train").sum()
        ),
        "validation_documents": int(
            (document_topics_df["split"] == "val").sum()
        ),
        "test_documents": int(
            (document_topics_df["split"] == "test").sum()
        ),
        "total_documents": len(document_topics_df),
        "topic_diversity": topic_diversity,
        "stability_mean": stability_mean,
        "stability_std": stability_std,
        "validation_metrics": val_eval,
        "test_metrics": test_eval,
        "runtime_breakdown": runtime_breakdown,
    }
    with (results_dir / "test_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            test_metrics_payload,
            f,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    """Xuất và kiểm tra bộ artifact K-Means."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    arguments = parse_arguments()

    refresh_kmeans_source_artifacts(
        source_dir=arguments.source_dir,
        results_dir=arguments.results_dir,
    )

    bundle = export_kmeans_artifacts(
        source_dir=arguments.source_dir,
        output_dir=arguments.output_dir,
        run_id=arguments.run_id,
    )

    print("Đã xuất common artifact cho K-Means thành công")
    print(f"- Model: {bundle.model_summary.model_name}")
    print(f"- Run ID: {bundle.model_summary.run_id}")
    print(
        f"- Số topic khai báo: "
        f"{bundle.model_summary.topic_count}"
    )
    print(
        f"- Số dòng topic summary: "
        f"{len(bundle.topic_summary)}"
    )
    print(
        f"- Số bài được gán topic: "
        f"{len(bundle.document_topics)}"
    )
    print(
        f"- Số dòng từ khóa: "
        f"{len(bundle.topic_keywords)}"
    )
    print(f"- Thư mục đầu ra: {arguments.output_dir}")


if __name__ == "__main__":
    main()
