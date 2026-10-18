"""Xuất kết quả mô hình K-Means sang common artifact schema."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.artifacts.adapters import (
    export_kmeans_artifacts,
)


DEFAULT_SOURCE_DIR = Path("artifacts/kmeans")
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/common/kmeans/kmeans_baseline"
)


def parse_arguments() -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description=(
            "Chuyển kết quả K-Means "
            "sang common artifact schema."
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
        "--run-id",
        type=str,
        default="kmeans_baseline",
        help="Định danh run_id cho bộ artifact.",
    )

    return parser.parse_args()


def main() -> None:
    """Xuất và kiểm tra bộ artifact K-Means."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    arguments = parse_arguments()

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

