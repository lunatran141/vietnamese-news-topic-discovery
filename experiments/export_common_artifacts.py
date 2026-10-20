"""Xuất kết quả mô hình sang common artifact schema."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.artifacts.adapters import (
    export_bertopic_artifacts,
    export_nmf_artifacts,
)


DEFAULT_SOURCE_DIR = Path(
    "results/bertopic_experiments/"
    "e1_vietnamese_sbert"
)

DEFAULT_OUTPUT_DIR = Path(
    "artifacts/common/bertopic/"
    "e1_vietnamese_sbert"
)

DEFAULT_NMF_SOURCE_DIR = Path(
    "artifacts/nmf/nmf_p3_k10_seed42_v1"
)

DEFAULT_NMF_OUTPUT_DIR = Path(
    "artifacts/common/nmf/nmf_p3_k10_seed42_v1"
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
        "--model",
        choices=["bertopic", "nmf"],
        default="bertopic",
        help="Phương pháp cần xuất sang common artifact schema.",
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Thư mục kết quả BERTopic đầu vào.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Thư mục common artifact đầu ra.",
    )

    return parser.parse_args()


def main() -> None:
    """Xuất và kiểm tra bộ artifact BERTopic."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    arguments = parse_arguments()
    if arguments.model == "nmf":
        source_dir = arguments.source_dir or DEFAULT_NMF_SOURCE_DIR
        output_dir = arguments.output_dir or DEFAULT_NMF_OUTPUT_DIR
        bundle = export_nmf_artifacts(
            source_dir=source_dir,
            output_dir=output_dir,
        )
    else:
        source_dir = arguments.source_dir or DEFAULT_SOURCE_DIR
        output_dir = arguments.output_dir or DEFAULT_OUTPUT_DIR
        bundle = export_bertopic_artifacts(
            source_dir=source_dir,
            output_dir=output_dir,
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
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
