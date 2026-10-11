"""
Phân chia tập dữ liệu huấn luyện, kiểm định và kiểm thử (Train / Validation / Test).
Nhiệm vụ D05: Dataset Splits.

Nguyên tắc cốt lõi:
1. Chia tập một lần sau dedup và trước khi huấn luyện. Mọi mô hình dùng cùng split IDs.
2. Tránh rò rỉ dữ liệu bằng cách gom các bài viết thuộc cùng sự kiện
   hoặc near-duplicate vào cùng một nhóm.
3. Phân tầng theo chuyên mục với tỷ lệ 70% / 15% / 15%.
4. Giám sát phân bố nguồn báo và phân bố thời gian.
5. Chỉ lưu danh sách article_id vào thư mục data/splits/, không sao chép nhiều phiên bản dữ liệu văn bản.
6. Cố định hạt giống ngẫu nhiên (random_seed = 42).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("build_splits")

DEFAULT_CONFIG_PATH = Path("configs/split.yaml")
DEFAULT_INPUT_CSV = Path("data/interim/news_articles_deduplicated.csv")
DEFAULT_CANDIDATES_CSV = Path("results/dedup_data/dedup_candidates_review.csv")


def load_split_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Đọc cấu hình phân chia tập dữ liệu từ file YAML."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình split: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("split", {})


def build_event_clusters(
    candidates_path: Path | str,
    valid_ids: Set[str],
    content_thresh: float = 0.60,
    title_thresh: float = 0.60,
) -> Dict[str, str]:
    c_path = Path(candidates_path)
    adj: Dict[str, Set[str]] = defaultdict(set)

    if c_path.is_file():
        df_cand = pd.read_csv(c_path)
        # Chỉ xét các bài còn tồn tại trong bộ dữ liệu sau khử trùng lặp
        mask_valid = df_cand["article_id_1"].isin(valid_ids) & df_cand["article_id_2"].isin(valid_ids)
        mask_sim = (df_cand["content_similarity"] >= content_thresh) | (df_cand["title_similarity"] >= title_thresh)
        filtered_pairs = df_cand[mask_valid & mask_sim]

        for _, row in filtered_pairs.iterrows():
            u = str(row["article_id_1"])
            v = str(row["article_id_2"])
            adj[u].add(v)
            adj[v].add(u)
        logger.info("Đã tìm thấy %d cặp bài viết tương đồng cao để gom cụm sự kiện.", len(filtered_pairs))
    else:
        logger.warning("Không tìm thấy file ứng viên dedup: %s. Chuyển sang chia nhóm độc lập.", c_path)

    # Tìm các thành phần liên thông bằng thuật toán BFS
    visited: Set[str] = set()
    group_map: Dict[str, str] = {}
    group_idx = 0
    multi_count = 0

    for node in sorted(valid_ids):
        if node not in visited:
            comp: List[str] = []
            queue = [node]
            visited.add(node)
            for curr in queue:
                comp.append(curr)
                for neighbor in sorted(adj[curr]):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            group_name = f"event_grp_{group_idx:05d}"
            if len(comp) > 1:
                multi_count += 1
            for nid in comp:
                group_map[nid] = group_name
            group_idx += 1

    logger.info("Đã tạo %d nhóm (trong đó có %d cụm đa bài viết cùng sự kiện).", group_idx, multi_count)
    return group_map

def create_stratified_group_splits(
    df: pd.DataFrame,
    group_map: Dict[str, str],
    stratify_col: str = "category_gold",
    id_col: str = "article_id",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-4:
        raise ValueError(f"Tổng tỷ lệ phải bằng 1.0, hiện tại là {total_ratio}")

    df_work = df.copy()
    df_work["group_id"] = df_work[id_col].map(group_map)

    # Chọn nhãn đại diện cho cả nhóm để đảm bảo tính phân tầng chuẩn
    group_label_map = df_work.groupby("group_id")[stratify_col].agg(lambda s: s.mode()[0])
    df_work["group_stratify_label"] = df_work["group_id"].map(group_label_map)

    # Tách 70% Train và 30% Temp (10 folds)
    sgkf1 = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=random_state)
    fold_assignments = np.zeros(len(df_work), dtype=int)
    for fold, (_, test_idx) in enumerate(sgkf1.split(df_work, df_work["group_stratify_label"], df_work["group_id"])):
        fold_assignments[test_idx] = fold

    # Folds 0-6 làm Train, Folds 7-9 làm Temp
    train_mask = np.isin(fold_assignments, [0, 1, 2, 3, 4, 5, 6])
    temp_mask = ~train_mask

    df_temp = df_work[temp_mask].copy()

    # Tách Temp thành Validation và Test 
    sgkf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=random_state)
    temp_splits = list(sgkf2.split(df_temp, df_temp["group_stratify_label"], df_temp["group_id"]))

    val_indices = df_temp.iloc[temp_splits[0][0]].index
    test_indices = df_temp.iloc[temp_splits[0][1]].index

    df_work["split"] = "train"
    df_work.loc[val_indices, "split"] = "val"
    df_work.loc[test_indices, "split"] = "test"

    # Kiểm tra rò rỉ nhóm giữa các tập
    train_grps = set(df_work[df_work["split"] == "train"]["group_id"])
    val_grps = set(df_work[df_work["split"] == "val"]["group_id"])
    test_grps = set(df_work[df_work["split"] == "test"]["group_id"])

    assert len(train_grps & val_grps) == 0, "Phát hiện rò rỉ nhóm sự kiện giữa Train và Val!"
    assert len(train_grps & test_grps) == 0, "Phát hiện rò rỉ nhóm sự kiện giữa Train và Test!"
    assert len(val_grps & test_grps) == 0, "Phát hiện rò rỉ nhóm sự kiện giữa Val và Test!"

    logger.info(
        "Phân chia hoàn tất thành công: Train=%d (%.2f%%), Val=%d (%.2f%%), Test=%d (%.2f%%)",
        len(df_work[df_work["split"] == "train"]),
        len(df_work[df_work["split"] == "train"]) / len(df_work) * 100,
        len(df_work[df_work["split"] == "val"]),
        len(df_work[df_work["split"] == "val"]) / len(df_work) * 100,
        len(df_work[df_work["split"] == "test"]),
        len(df_work[df_work["split"] == "test"]) / len(df_work) * 100,
    )

    return df_work


def save_split_files(
    df_splits: pd.DataFrame,
    output_dir: Path,
    id_col: str = "article_id",
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: Dict[str, Path] = {}

    train_ids = df_splits[df_splits["split"] == "train"][[id_col]].sort_values(by=id_col).reset_index(drop=True)
    val_ids = df_splits[df_splits["split"] == "val"][[id_col]].sort_values(by=id_col).reset_index(drop=True)
    test_ids = df_splits[df_splits["split"] == "test"][[id_col]].sort_values(by=id_col).reset_index(drop=True)

    train_path = output_dir / "train_ids.csv"
    val_path = output_dir / "validation_ids.csv"
    test_path = output_dir / "test_ids.csv"
    manifest_path = output_dir / "split_manifest.csv"

    train_ids.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_ids.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_ids.to_csv(test_path, index=False, encoding="utf-8-sig")

    saved_files["train"] = train_path
    saved_files["val"] = val_path
    saved_files["test"] = test_path

    # Lưu bảng chỉ mục tổng hợp đầy đủ để phục vụ tra cứu phân tích
    meta_cols = [id_col, "split", "category_gold", "source", "published_at", "group_id"]
    cols_to_save = [c for c in meta_cols if c in df_splits.columns]
    df_manifest = df_splits[cols_to_save].sort_values(by=id_col).reset_index(drop=True)
    df_manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    saved_files["manifest"] = manifest_path

    logger.info("Đã lưu các file split IDs: train=%s, val=%s, test=%s", train_path.name, val_path.name, test_path.name)
    return saved_files


def generate_monitoring_reports(
    df_splits: pd.DataFrame,
    results_dir: Path,
) -> None:
    """Tạo các báo cáo giám sát phân bố chuyên mục, nguồn báo và thời gian."""
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Báo cáo tổng hợp số lượng và tỷ lệ
    summary_df = pd.DataFrame({
        "split": ["train", "val", "test", "total"],
        "count": [
            (df_splits["split"] == "train").sum(),
            (df_splits["split"] == "val").sum(),
            (df_splits["split"] == "test").sum(),
            len(df_splits),
        ],
        "percentage": [
            round((df_splits["split"] == "train").mean() * 100, 2),
            round((df_splits["split"] == "val").mean() * 100, 2),
            round((df_splits["split"] == "test").mean() * 100, 2),
            100.0,
        ],
    })
    summary_df.to_csv(results_dir / "split_summary.csv", index=False, encoding="utf-8-sig")

    # Báo cáo phân tầng chuyên mục
    cat_counts = pd.crosstab(df_splits["category_gold"], df_splits["split"], margins=True)
    cat_pct = (pd.crosstab(df_splits["category_gold"], df_splits["split"], normalize="columns") * 100).round(2)
    cat_combined = cat_counts.copy()
    for col in ["train", "val", "test"]:
        cat_combined[f"{col}_pct"] = cat_pct[col]
    cat_combined.to_csv(results_dir / "split_category_distribution.csv", encoding="utf-8-sig")

    # Báo cáo phân bố nguồn báo
    source_counts = pd.crosstab(df_splits["source"], df_splits["split"], margins=True)
    source_pct = (pd.crosstab(df_splits["source"], df_splits["split"], normalize="columns") * 100).round(2)
    source_combined = source_counts.copy()
    for col in ["train", "val", "test"]:
        source_combined[f"{col}_pct"] = source_pct[col]
    source_combined.to_csv(results_dir / "split_source_distribution.csv", encoding="utf-8-sig")

    # Báo cáo phân bố thời gian theo tuần
    df_splits_copy = df_splits.copy()
    df_splits_copy["published_at"] = pd.to_datetime(df_splits_copy["published_at"], errors="coerce")
    df_splits_copy["week"] = df_splits_copy["published_at"].dt.isocalendar().week
    week_counts = pd.crosstab(df_splits_copy["week"], df_splits_copy["split"], margins=True)
    week_pct = (pd.crosstab(df_splits_copy["week"], df_splits_copy["split"], normalize="columns") * 100).round(2)
    week_combined = week_counts.copy()
    for col in ["train", "val", "test"]:
        week_combined[f"{col}_pct"] = week_pct[col]
    week_combined.to_csv(results_dir / "split_temporal_distribution.csv", encoding="utf-8-sig")

    logger.info("Đã xuất 4 bảng báo cáo giám sát vào: %s", results_dir)

    # Vẽ biểu đồ trực quan hóa
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Cấu hình phong cách và font
        plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        sns.set_theme(style="whitegrid")

        # Biểu đồ 1: Phân tầng chuyên mục category_gold
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        cat_df_plot = df_splits.groupby(["category_gold", "split"]).size().reset_index(name="count")
        # Tính tỷ lệ trong từng split
        split_totals = df_splits["split"].value_counts().to_dict()
        cat_df_plot["pct"] = cat_df_plot.apply(lambda r: r["count"] / split_totals[r["split"]] * 100, axis=1)

        palette = {"train": "#2b5c8f", "val": "#d95f02", "test": "#2ca02c"}
        sns.barplot(data=cat_df_plot, x="category_gold", y="pct", hue="split", palette=palette, ax=ax)
        ax.set_title("Phân bố chuyên mục trên các tập dữ liệu", fontsize=13, weight="bold")
        ax.set_xlabel("Chuyên mục", fontsize=11)
        ax.set_ylabel("Tỷ lệ phần trăm (%)", fontsize=11)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
        ax.legend(title="Tập dữ liệu")
        fig.tight_layout()
        fig.savefig(figures_dir / "d05_category_stratification.png")
        plt.close(fig)

        # Biểu đồ 2: Phân bố nguồn báo (source)
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        source_df_plot = df_splits.groupby(["source", "split"]).size().reset_index(name="count")
        source_df_plot["pct"] = source_df_plot.apply(lambda r: r["count"] / split_totals[r["split"]] * 100, axis=1)

        sns.barplot(data=source_df_plot, x="source", y="pct", hue="split", palette=palette, ax=ax)
        ax.set_title("Phân bố nguồn báo trên các tập dữ liệu", fontsize=13, weight="bold")
        ax.set_xlabel("Nguồn báo", fontsize=11)
        ax.set_ylabel("Tỷ lệ phần trăm (%)", fontsize=11)
        ax.legend(title="Tập dữ liệu")
        fig.tight_layout()
        fig.savefig(figures_dir / "d05_source_distribution.png")
        plt.close(fig)

        # Biểu đồ 3: Phân bố theo chuỗi thời gian (timeline ngày)
        fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
        df_splits_copy["day"] = df_splits_copy["published_at"].dt.date
        day_df_plot = df_splits_copy.groupby(["day", "split"]).size().reset_index(name="count")
        sns.lineplot(data=day_df_plot, x="day", y="count", hue="split", palette=palette, marker="o", ax=ax)
        ax.set_title("Phân bố số lượng bài viết theo ngày trong Tháng 9/2026", fontsize=13, weight="bold")
        ax.set_xlabel("Ngày xuất bản", fontsize=11)
        ax.set_ylabel("Số lượng bài", fontsize=11)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Tập dữ liệu")
        fig.tight_layout()
        fig.savefig(figures_dir / "d05_temporal_timeline.png")
        plt.close(fig)

        logger.info("Đã vẽ 3 biểu đồ trực quan hóa vào: %s", figures_dir)
    except Exception as exc:
        logger.warning("Không thể xuất biểu đồ trực quan: %s", exc)


def parse_args() -> argparse.Namespace:
    """Xử lý tham số dòng lệnh CLI."""
    parser = argparse.ArgumentParser(description="D05: Phân chia tập Train / Validation / Test")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Đường dẫn file cấu hình split (mặc định configs/split.yaml)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_CSV),
        help="Đường dẫn file dữ liệu sạch sau dedup",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/splits",
        help="Thư mục lưu các file split IDs",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/splits_data",
        help="Thư mục lưu báo cáo giám sát phân bố",
    )
    return parser.parse_args()


def main():
    """Hàm điều phối thực thi chính của pipeline D05."""
    args = parse_args()
    config = load_split_config(args.config)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)

    if not input_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu đầu vào: {input_path}")

    logger.info("Đang nạp dữ liệu sạch từ: %s", input_path)
    df = pd.read_csv(input_path)
    logger.info("Dữ liệu nạp thành công: %d bài viết", len(df))

    # Đọc tham số cấu hình
    train_ratio = float(config.get("train_ratio", 0.70))
    val_ratio = float(config.get("val_ratio", 0.15))
    test_ratio = float(config.get("test_ratio", 0.15))
    seed = int(config.get("random_seed", 42))
    stratify_col = config.get("stratify_column", "category_gold")
    id_col = config.get("id_column", "article_id")

    # Gom cụm sự kiện chống rò rỉ dữ liệu (Event Clustering)
    event_cfg = config.get("event_grouping", {})
    if event_cfg.get("enabled", True):
        cand_file = Path(event_cfg.get("candidates_file", str(DEFAULT_CANDIDATES_CSV)))
        c_thresh = float(event_cfg.get("content_similarity_threshold", 0.60))
        t_thresh = float(event_cfg.get("title_similarity_threshold", 0.60))
        valid_ids = set(df[id_col].astype(str))
        group_map = build_event_clusters(cand_file, valid_ids, c_thresh, t_thresh)
    else:
        group_map = {str(aid): str(aid) for aid in df[id_col]}

    # Phân tầng Stratified Group Split
    df_splits = create_stratified_group_splits(
        df=df,
        group_map=group_map,
        stratify_col=stratify_col,
        id_col=id_col,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=seed,
    )

    # Lưu 3 file IDs bắt buộc theo đặc tả
    save_split_files(df_splits, output_dir=output_dir, id_col=id_col)

    # Sinh báo cáo giám sát phân bố
    generate_monitoring_reports(df_splits, results_dir=results_dir)

    logger.info("Hoàn tất pipeline D05 Chia Train / Validation / Test thành công!")


if __name__ == "__main__":
    main()
