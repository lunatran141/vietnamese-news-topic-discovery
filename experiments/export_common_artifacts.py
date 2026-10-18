"""Xuất kết quả mô hình sang common artifact schema."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.artifacts.adapters import (
    export_bertopic_artifacts,
)


DEFAULT_SOURCE_DIR = Path(
    "results/bertopic_experiments/"
    "e1_vietnamese_sbert"
)

DEFAULT_OUTPUT_DIR = Path(
    "artifacts/common/bertopic/"
    "e1_vietnamese_sbert"
)


def parse_arguments() -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description=(
            "Chuyển kết quả BERTopic "
            "sang common artifact schema."
        )
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Thư mục kết quả BERTopic đầu vào.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Thư mục common artifact đầu ra.",
    )

    return parser.parse_args()


def main() -> None:
    """Xuất và kiểm tra bộ artifact BERTopic."""
    arguments = parse_arguments()

    bundle = export_bertopic_artifacts(
        source_dir=arguments.source_dir,
        output_dir=arguments.output_dir,
    )

    print("Đã xuất common artifact thành công")
    print(
        f"- Model: {bundle.model_summary.model_name}"
    )
    print(
        f"- Run ID: {bundle.model_summary.run_id}"
    )
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
    print(
        f"- Thư mục đầu ra: "
        f"{arguments.output_dir}"
    )


if __name__ == "__main__":
    main()