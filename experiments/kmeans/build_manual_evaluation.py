"""Sinh bảng đánh giá thủ công cho K-Means"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import numpy as np
import pandas as pd
import yaml

from src.features.tfidf import load_tfidf_artifacts
from src.models.kmeans import extract_cluster_keywords, predict_kmeans, train_kmeans

DRAFT_STATUS = "draft_assistant_pending_confirmation"
EMPTY_STATUS = "not_reviewed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sinh bảng đánh giá thủ công K-Means")
    parser.add_argument("--config", default="configs/kmeans.yaml")
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[7, 9, 11],
        help="Các giá trị K cần đánh giá thủ công",
    )
    parser.add_argument(
        "--drafts",
        default="results/kmeans/kmeans_manual_evaluation_drafts.json",
        help="File JSON nháp nhãn thủ công, khóa dạng 'k:cluster_id'",
    )
    parser.add_argument(
        "--output",
        default="results/kmeans/kmeans_manual_evaluation.csv",
    )
    parser.add_argument("--top-docs", type=int, default=5)
    parser.add_argument("--random-docs", type=int, default=5)
    return parser.parse_args()


def load_drafts(path: Path) -> dict[str, dict[str, Any]]:
    """Đọc nháp nhãn thủ công; trả về dict rỗng nếu chưa có file."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def shorten(text: str, limit: int = 110) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_rows_for_k(
    k: int,
    clustering_cfg: dict[str, Any],
    train_matrix: Any,
    val_matrix: Any,
    train_ids: list[str],
    val_ids: list[str],
    feature_names: list[str],
    meta: pd.DataFrame,
    drafts: dict[str, dict[str, Any]],
    top_docs: int,
    random_docs: int,
) -> list[dict[str, Any]]:
    """Tạo các hàng đánh giá thủ công cho một giá trị K."""
    model = train_kmeans(
        train_matrix,
        n_clusters=k,
        init=clustering_cfg.get("init", "k-means++"),
        n_init=clustering_cfg.get("n_init", 10),
        max_iter=clustering_cfg.get("max_iter", 300),
        random_state=clustering_cfg.get("random_state", 42),
    )
    train_clusters, train_dists = predict_kmeans(model, train_matrix)
    val_clusters, _ = predict_kmeans(model, val_matrix)
    keywords = extract_cluster_keywords(model, feature_names, top_n=10)

    title_lookup = meta.set_index("article_id")["title"].to_dict()
    gold_lookup = meta.set_index("article_id")["category_gold"].to_dict()
    rng = np.random.RandomState(clustering_cfg.get("random_state", 42))

    rows: list[dict[str, Any]] = []
    for cluster_id in range(k):
        train_idx = np.where(train_clusters == cluster_id)[0]
        val_idx = np.where(val_clusters == cluster_id)[0]

        # Bài gần tâm nhất lấy từ Train
        nearest = train_idx[np.argsort(train_dists[train_idx])[:top_docs]]
        nearest_titles = [shorten(title_lookup.get(train_ids[i], "")) for i in nearest]

        # Bài ngẫu nhiên lấy từ Val
        take = min(random_docs, len(val_idx))
        random_idx = rng.choice(val_idx, size=take, replace=False) if take else []
        random_titles = [shorten(title_lookup.get(val_ids[i], "")) for i in random_idx]

        val_gold = pd.Series([gold_lookup.get(val_ids[i], "") for i in val_idx])
        if len(val_gold):
            gold_counts = val_gold.value_counts()
            dominant_gold = gold_counts.index[0]
            dominant_share = round(float(gold_counts.iloc[0] / len(val_gold)), 3)
            gold_breakdown = "; ".join(
                f"{name}={count}" for name, count in gold_counts.head(3).items()
            )
        else:
            dominant_gold, dominant_share, gold_breakdown = "", 0.0, ""

        draft = drafts.get(f"{k}:{cluster_id}", {})
        rows.append({
            "k": k,
            "cluster_id": cluster_id,
            "size_train": int(len(train_idx)),
            "size_val": int(len(val_idx)),
            "top_keywords": ", ".join(item["term"] for item in keywords[str(cluster_id)]),
            "nearest_titles_train": " || ".join(nearest_titles),
            "random_titles_val": " || ".join(random_titles),
            "dominant_gold_val": dominant_gold,
            "dominant_gold_share_val": dominant_share,
            "gold_top3_val": gold_breakdown,
            "manual_label": draft.get("manual_label", ""),
            "coherence_1_5": draft.get("coherence_1_5", ""),
            "is_mixed_topic": draft.get("is_mixed_topic", ""),
            "notes": draft.get("notes", ""),
            "review_status": draft.get(
                "review_status", DRAFT_STATUS if draft else EMPTY_STATUS
            ),
        })
    return rows


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    dataset_cfg = config["dataset"]
    clustering_cfg = config["clustering"]

    tfidf_dir = Path(dataset_cfg.get("tfidf_artifacts_dir", "artifacts/tfidf"))
    train_bundle = load_tfidf_artifacts(tfidf_dir, split="train")
    val_bundle = load_tfidf_artifacts(tfidf_dir, split="val")

    meta = pd.read_csv(
        dataset_cfg.get("processed_dataset_path", "data/processed/p2_segmented.csv"),
        usecols=["article_id", "title", "category_gold"],
    )
    meta["article_id"] = meta["article_id"].astype(str)

    drafts = load_drafts(Path(args.drafts))
    all_rows: list[dict[str, Any]] = []
    for k in args.k_values:
        print(f"Đang xử lý K = {k}...")
        all_rows.extend(
            build_rows_for_k(
                k,
                clustering_cfg,
                train_bundle["matrix"],
                val_bundle["matrix"],
                train_bundle["article_ids"],
                val_bundle["article_ids"],
                train_bundle["feature_names"],
                meta,
                drafts,
                args.top_docs,
                args.random_docs,
            )
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Đã lưu {len(all_rows)} hàng vào: {output_path}")


if __name__ == "__main__":
    main()

