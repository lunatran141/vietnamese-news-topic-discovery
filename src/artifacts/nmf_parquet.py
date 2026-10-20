"""Đọc Parquet NMF và chuẩn hóa topic ID cho common artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


_TOPIC_ID_PATTERN = re.compile(r"^NMF_(\d+)$")


def normalize_nmf_topic_id(value: object) -> int:
    """Chuyển ID NMF dạng ``NMF_XX`` về số nguyên; giữ ``-1`` cho outlier."""
    if pd.isna(value):
        raise ValueError("topic_id NMF không được để trống")
    if isinstance(value, (int, np.integer)):
        topic_id = int(value)
    elif isinstance(value, (float, np.floating)) and float(value).is_integer():
        topic_id = int(value)
    elif isinstance(value, str) and value == "-1":
        topic_id = -1
    elif isinstance(value, str):
        match = _TOPIC_ID_PATTERN.fullmatch(value)
        if match is None:
            raise ValueError(f"topic_id NMF không hợp lệ: {value}")
        topic_id = int(match.group(1))
    else:
        raise ValueError(f"topic_id NMF không hợp lệ: {value}")
    if topic_id < -1:
        raise ValueError(f"topic_id NMF không hợp lệ: {value}")
    return topic_id


def _share_columns(frame: pd.DataFrame) -> list[str]:
    """Trả các cột trọng số topic theo thứ tự chỉ số."""
    columns = [
        column
        for column in frame.columns
        if re.fullmatch(r"topic_\d+_share", str(column))
    ]
    return sorted(columns, key=lambda column: int(column.split("_")[1]))


def _normalize_assignment_frame(
    frame: pd.DataFrame,
    expected_splits: set[str],
    artifact_name: str,
) -> pd.DataFrame:
    """Chuẩn hóa một split Parquet thành assignments NMF dạng common."""
    if frame.empty:
        raise ValueError(f"Parquet NMF {artifact_name} không có dòng")
    if "article_id" not in frame or frame["article_id"].isna().any():
        raise ValueError(f"Parquet NMF {artifact_name} thiếu article_id")
    if frame["article_id"].astype(str).str.strip().eq("").any():
        raise ValueError(f"Parquet NMF {artifact_name} có article_id rỗng")
    if frame["article_id"].astype(str).duplicated().any():
        raise ValueError(f"Parquet NMF {artifact_name} có article_id trùng")

    result = frame.copy()
    result["article_id"] = result["article_id"].astype(str)
    if "split" in result:
        actual_splits = set(result["split"].dropna().astype(str))
        if actual_splits != expected_splits:
            raise ValueError(
                "Split trong Parquet không khớp "
                f"{artifact_name}: {sorted(actual_splits)}"
            )
    if "split" not in result:
        if len(expected_splits) != 1:
            raise ValueError(f"Parquet {artifact_name} thiếu cột split")
        result["split"] = next(iter(expected_splits))

    shares = _share_columns(result)
    if not shares:
        raise ValueError(f"Parquet NMF {artifact_name} thiếu topic share columns")
    share_values = result[shares].apply(pd.to_numeric, errors="raise").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(share_values).all() or (share_values < 0).any():
        raise ValueError(f"Topic shares của {artifact_name} phải hữu hạn và không âm")
    if (share_values > 1 + 1e-9).any():
        raise ValueError(f"Topic shares của {artifact_name} vượt quá 1")
    share_sums = share_values.sum(axis=1)
    if not np.allclose(share_sums, 1.0, atol=1e-6):
        raise ValueError(f"Topic shares của {artifact_name} phải cộng thành 1")
    result[shares] = share_values
    derived_primary = np.asarray(
        [int(column.split("_")[1]) for column in shares], dtype=np.int64
    )[share_values.argmax(axis=1)]

    if "primary_topic_id" in result:
        primary_ids = result["primary_topic_id"].map(normalize_nmf_topic_id)
        if not np.array_equal(primary_ids.to_numpy(dtype=np.int64), derived_primary):
            raise ValueError(
                f"Primary topic ID không khớp topic shares của {artifact_name}"
            )
    result["topic_id"] = derived_primary

    if "primary_topic_share" in result:
        primary_shares = pd.to_numeric(result["primary_topic_share"], errors="raise")
        if not np.allclose(primary_shares, share_values.max(axis=1), atol=1e-6):
            raise ValueError(
                f"Primary topic share không khớp topic shares của {artifact_name}"
            )
    else:
        result["primary_topic_share"] = share_values.max(axis=1)

    if "secondary_topic_id" in result:
        result["secondary_topic_id"] = result["secondary_topic_id"].map(
            normalize_nmf_topic_id
        )
    if "assigned_topic_ids" in result:
        def normalize_topic_list(value: object) -> str:
            try:
                values = json.loads(str(value))
            except json.JSONDecodeError as error:
                raise ValueError("assigned_topic_ids phải là JSON list") from error
            if not isinstance(values, list):
                raise ValueError("assigned_topic_ids phải là JSON list")
            return json.dumps(
                [normalize_nmf_topic_id(topic_id) for topic_id in values]
            )

        result["assigned_topic_ids"] = result["assigned_topic_ids"].map(
            normalize_topic_list
        )

    result["topic_probability"] = pd.Series(
        pd.NA, index=result.index, dtype="Float64"
    )
    result["is_outlier"] = result["topic_id"].eq(-1)
    return result


def read_nmf_parquet_assignments(run_dir: str | Path) -> pd.DataFrame:
    """Đọc Parquet development/test và ghép assignments theo article_id."""
    run_dir = Path(run_dir)
    development_path = run_dir / "development_document_topics.parquet"
    test_path = run_dir / "document_topics.parquet"
    missing = [
        path.name
        for path in (development_path, test_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Thiếu Parquet NMF bắt buộc: " + ", ".join(missing)
        )

    development = _normalize_assignment_frame(
        pd.read_parquet(development_path), {"train", "val"}, "development"
    )
    test = _normalize_assignment_frame(
        pd.read_parquet(test_path), {"test"}, "test"
    )
    assignments = pd.concat([development, test], ignore_index=True, sort=False)
    if assignments["article_id"].duplicated().any():
        raise ValueError("article_id bị trùng giữa Parquet development và test")
    if assignments["topic_id"].lt(-1).any():
        raise ValueError("topic_id chuẩn hóa không được nhỏ hơn -1")
    return assignments
