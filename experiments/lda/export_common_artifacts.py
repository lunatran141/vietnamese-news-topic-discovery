
"""Export frozen LDA K=15 using the common artifact adapter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.artifacts.adapters import export_lda_artifacts


DEFAULT_SOURCE_DIR = Path("artifacts/lda/final")
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/common/lda/lda_k15_seed42"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export LDA K=15 to common artifact schema."
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_arguments()

    bundle = export_lda_artifacts(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
    )

    print("LDA common artifact export completed.")
    print("Model:", bundle.model_summary.model_name)
    print("Run ID:", bundle.model_summary.run_id)
    print("Topics:", bundle.model_summary.topic_count)
    print("Documents:", len(bundle.document_topics))
    print("Keywords:", len(bundle.topic_keywords))
    print("Output:", args.output_dir)


if __name__ == "__main__":
    main()
