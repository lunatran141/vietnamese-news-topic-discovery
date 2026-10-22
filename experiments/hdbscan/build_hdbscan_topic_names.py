"""Kiểm tra phiếu tên HDBSCAN và xuất mapping cho giao diện."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "minilm_title_content_eom_15_5"

BUNDLE = ROOT / "artifacts/common/hdbscan" / RUN_ID
OUTPUT = ROOT / "results/hdbscan/topic_naming"
REVIEW_PATH = OUTPUT / "topic_name_review.csv"
EXAMPLES_PATH = OUTPUT / "topic_name_examples.csv"
MAPPING_PATH = OUTPUT / f"{RUN_ID}_display_mapping.json"
COMMON_MAPPING_PATH = BUNDLE / "topic_display_mapping.json"
GROUPS_PATH = BUNDLE / "topic_display_groups.csv"

def export_mapping() -> None:
    if not REVIEW_PATH.is_file():
        raise FileNotFoundError(f"Thiếu phiếu: {REVIEW_PATH}")

    summary = pd.read_csv(BUNDLE / "topic_summary.csv")
    review = pd.read_csv(
        REVIEW_PATH,
        encoding="utf-8-sig",
        keep_default_na=False,
    )

    expected_ids = set(summary["topic_id"].astype(int)) - {-1}
    actual_ids = set(review["topic_id"].astype(int))

    if len(expected_ids) != 101:
        raise ValueError(
            f"Artifact hiện có {len(expected_ids)} cụm, dự kiến 101"
        )
    if review["topic_id"].duplicated().any():
        raise ValueError("Phiếu có topic_id trùng")
    if actual_ids != expected_ids:
        raise ValueError("Phiếu không chứa đúng 101 topic_id của run")
    if not review["review_status"].str.strip().eq("approved").all():
        raise ValueError("Còn cụm chưa có review_status=approved")

    for column in ("suggest_label", "display_name"):
        if column not in review.columns:
            raise ValueError(f"Phiếu thiếu cột {column}")
        if review[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Có cụm chưa điền {column}")

    mapping = {
        str(int(row.topic_id)): {
            "suggest_label": str(row.suggest_label).strip(),
            "display_name": str(row.display_name).strip(),
            "display_group_id": None,
        }
        for row in review.itertuples(index=False)
    }

    if GROUPS_PATH.is_file():
        groups = pd.read_csv(
            GROUPS_PATH,
            encoding="utf-8-sig",
            keep_default_na=False,
            dtype=str,
        )

        required = {"topic_id", "display_group_id", "reason"}
        if not required.issubset(groups.columns):
            raise ValueError(f"File nhóm cần các cột: {sorted(required)}")

        groups = groups.copy()
        for column in required:
            groups[column] = groups[column].str.strip()

        if not groups["topic_id"].str.fullmatch(r"\d+").all():
            raise ValueError("Có topic_id không hợp lệ trong file nhóm")

        groups["topic_id"] = groups["topic_id"].astype(int)

        if groups["topic_id"].duplicated().any():
            raise ValueError("Một topic_id xuất hiện nhiều lần")

        if not set(groups["topic_id"]).issubset(expected_ids):
            raise ValueError("File nhóm chứa topic_id ngoài 101 cụm")

        if groups["display_group_id"].eq("").any():
            raise ValueError("Có tên nhóm rỗng")

        if groups["reason"].eq("").any():
            raise ValueError("Có nhóm chưa ghi lý do")

        group_sizes = groups.groupby("display_group_id").size()
        if group_sizes.lt(2).any():
            raise ValueError("Mỗi nhóm cần ít nhất hai cụm")

        for row in groups.itertuples(index=False):
            mapping[str(row.topic_id)]["display_group_id"] = (
                row.display_group_id
            )

    payload = {
        "run_id": RUN_ID,
        "schema_version": "1.0",
        "mapping": mapping,
    }

    if not BUNDLE.is_dir():
        raise FileNotFoundError(f"Thiếu common artifact: {BUNDLE}")

    output_text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    ) + "\n"

    OUTPUT.mkdir(parents=True, exist_ok=True)
    MAPPING_PATH.write_text(output_text, encoding="utf-8")
    COMMON_MAPPING_PATH.write_text(output_text, encoding="utf-8")

    if MAPPING_PATH.read_bytes() != COMMON_MAPPING_PATH.read_bytes():
        raise RuntimeError("Hai file mapping HDBSCAN không khớp")

    print(f"Đã xuất {len(mapping)} tên:")
    print(MAPPING_PATH)
    print(COMMON_MAPPING_PATH)

def check() -> None:
    if not EXAMPLES_PATH.is_file():
        raise FileNotFoundError(f"Thiếu bài ví dụ: {EXAMPLES_PATH}")

    examples = pd.read_csv(EXAMPLES_PATH, encoding="utf-8-sig")
    review = pd.read_csv(REVIEW_PATH, encoding="utf-8-sig")

    if len(review) != 101:
        raise ValueError("Phiếu phải có 101 cụm")
    if set(examples["topic_id"]) != set(review["topic_id"]):
        raise ValueError("Bài ví dụ không bao phủ đủ 101 cụm")

    print(f"Phiếu: {len(review)} cụm")
    print(f"Bài ví dụ: {len(examples)} bài")
    print("Dữ liệu đặt tên hợp lệ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export",
        action="store_true",
        help="Xuất lại JSON từ 101 tên trong phiếu",
    )
    args = parser.parse_args()

    check()
    if args.export:
        export_mapping()

if __name__ == "__main__":
    main()