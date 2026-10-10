"""
Khung sườn phân chia tập dữ liệu huấn luyện / kiểm định / kiểm thử.
Module: src/preprocessing/build_splits.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

# Cấu hình mã hóa đầu ra UTF-8 an toàn trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("build_splits")


def create_stratified_split_indices(
    df: pd.DataFrame,
    stratify_col: str = "category_gold",
    id_col: str = "article_id",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Tạo bảng chỉ mục ánh xạ article_id -> split ('train', 'val', 'test').
    
    Phân tầng dựa trên cột stratify_col để giữ nguyên tỷ lệ nhãn chủ đề.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-5:
        raise ValueError(f"Tổng các tỷ lệ phải bằng 1.0, hiện tại là {total_ratio}")

    df_ids = df[[id_col, stratify_col]].drop_duplicates().copy()

    # Chia tách tập Train và tập tạm thời (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_ids, temp_ids = train_test_split(
        df_ids,
        test_size=temp_ratio,
        stratify=df_ids[stratify_col],
        random_state=random_state,
    )

    # Chia tách tập tạm thời thành Val và Test theo tỷ lệ tương ứng
    test_share_of_temp = test_ratio / temp_ratio
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=test_share_of_temp,
        stratify=temp_ids[stratify_col],
        random_state=random_state,
    )

    # Đánh dấu phân vùng
    train_ids = train_ids.copy()
    train_ids["split"] = "train"
    val_ids = val_ids.copy()
    val_ids["split"] = "val"
    test_ids = test_ids.copy()
    test_ids["split"] = "test"

    manifest = pd.concat([train_ids, val_ids, test_ids], ignore_index=True)
    manifest = manifest.sort_values(by=id_col).reset_index(drop=True)

    logger.info("Đã phân chia chỉ mục: Train=%d, Val=%d, Test=%d", len(train_ids), len(val_ids), len(test_ids))
    return manifest[[id_col, "split"]]


def apply_splits_to_variant(
    df: pd.DataFrame,
    manifest: pd.DataFrame,
    id_col: str = "article_id",
) -> Dict[str, pd.DataFrame]:
    """Áp dụng chỉ mục phân chia vào một DataFrame biến thể (P0, P1, P2 hoặc P3)."""
    merged = df.merge(manifest, on=id_col, how="inner")
    splits = {
        "train": merged[merged["split"] == "train"].drop(columns=["split"]),
        "val": merged[merged["split"] == "val"].drop(columns=["split"]),
        "test": merged[merged["split"] == "test"].drop(columns=["split"]),
    }
    return splits


def parse_args() -> argparse.Namespace:
    """Xử lý tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description="Phân chia dữ liệu tiền xử lý thành Train / Val / Test")
    parser.add_argument(
        "--variant",
        type=str,
        default="p2_segmented",
        choices=["p0_minimal", "p1_normalized", "p2_segmented", "p3_no_stopwords", "all"],
        help="Biến thể dữ liệu cần phân chia (hoặc 'all' cho toàn bộ biến thể)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/processed",
        help="Thư mục chứa các file biến thể đã tiền xử lý",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/splits",
        help="Thư mục lưu các tập phân chia",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Tỷ lệ tập Train (mặc định 0.70)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Tỷ lệ tập Validation (mặc định 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Tỷ lệ tập Test (mặc định 0.15)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Hạt giống ngẫu nhiên (mặc định 42)",
    )
    return parser.parse_args()


def main():
    """Hàm điều phối chính khi chạy CLI."""
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Đọc dữ liệu biến thể chuẩn làm căn cứ phân tầng
    ref_variant = "p2_segmented" if args.variant == "all" else args.variant
    ref_path = input_dir / f"{ref_variant}.csv"
    if not ref_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {ref_path}")

    logger.info("Đang đọc dữ liệu từ: %s", ref_path)
    df_ref = pd.read_csv(ref_path)

    # Sinh manifest phân tầng
    manifest = create_stratified_split_indices(
        df_ref,
        stratify_col="category_gold",
        id_col="article_id",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_seed,
    )

    manifest_path = output_dir / "split_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    logger.info("Đã lưu bảng ánh xạ chỉ mục phân vùng tại: %s", manifest_path)


if __name__ == "__main__":
    main()
