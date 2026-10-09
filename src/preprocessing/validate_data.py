"""
Module thẩm định dữ liệu thô (Task D01 — Data Validation Pipeline).

Chức năng:
    - Đọc file CSV gốc từ data/raw/news_articles.csv
    - Kiểm tra: schema, khóa chính, missing values, định dạng ngày,
      categorical domain, số dòng
    - Xuất file interim đã validate ra data/interim/

Cách dùng:
    # Validate toàn bộ, không giới hạn số dòng cụ thể
    python -m src.preprocessing.validate_data

    # Validate với kỳ vọng cụ thể
    python -m src.preprocessing.validate_data --expected-rows 9304 --year 2026 --month 9

    # Đổi đường dẫn input/output
    python -m src.preprocessing.validate_data --raw data/raw/custom.csv --out data/interim/custom_validated.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_CSV = ROOT / "data" / "raw" / "news_articles.csv"
DEFAULT_OUTPUT_CSV = ROOT / "data" / "interim" / "news_articles_validated.csv"

REQUIRED_COLS: List[str] = [
    "article_id", "title", "content", "published_at",
    "source", "url", "category_gold",
]

VALID_SOURCES: Set[str] = {"vnexpress", "tuoitre", "vietnamnet"}

VALID_CATS: Set[str] = {
    "Kinh doanh", "Khoa học công nghệ", "Thể thao", "Giáo dục",
    "Sức khỏe", "Pháp luật", "Giải trí",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validate_data")


def check_schema(df: pd.DataFrame, required_cols: List[str]) -> None:
    if list(df.columns) != required_cols:
        missing = set(required_cols) - set(df.columns)
        extra = set(df.columns) - set(required_cols)
        raise ValueError(
            f"Sai schema cột!\n"
            f"  Kỳ vọng: {required_cols}\n"
            f"  Thực tế: {list(df.columns)}\n"
            f"  Thiếu:   {sorted(missing) if missing else '(không)'}\n"
            f"  Thừa:    {sorted(extra) if extra else '(không)'}"
        )


def check_id_uniqueness(df: pd.DataFrame, id_col: str = "article_id") -> None:
    n_null = df[id_col].isna().sum()
    if n_null > 0:
        raise ValueError(f"Cột '{id_col}' chứa {n_null} giá trị NaN")

    n_empty = (df[id_col].astype(str).str.strip() == "").sum()
    if n_empty > 0:
        raise ValueError(f"Cột '{id_col}' chứa {n_empty} giá trị rỗng")

    n_dup = df[id_col].duplicated().sum()
    if n_dup > 0:
        raise ValueError(f"Cột '{id_col}' có {n_dup} giá trị trùng lặp")


def check_missing_values(df: pd.DataFrame, cols_to_check: List[str]) -> None:
    for col in cols_to_check:
        n_null = df[col].isna().sum()
        if n_null > 0:
            raise ValueError(f"Cột '{col}' chứa {n_null} giá trị NaN")

        n_empty = (df[col].astype(str).str.strip() == "").sum()
        if n_empty > 0:
            raise ValueError(f"Cột '{col}' chứa {n_empty} giá trị rỗng")


def check_datetime_range(
    df: pd.DataFrame,
    date_col: str = "published_at",
    year: Optional[int] = None,
    month: Optional[int] = None,
) -> pd.DataFrame:
    df_copy = df.copy()
    parsed = pd.to_datetime(df_copy[date_col], errors="coerce", utc=True)

    n_bad = parsed.isna().sum()
    if n_bad > 0:
        raise ValueError(
            f"Có {n_bad} bài viết không parse được cột '{date_col}'"
        )
    # Convert về giờ VN để so sánh tháng/năm
    parsed_vn = parsed.dt.tz_convert("Asia/Ho_Chi_Minh")
    if year is not None:
        n_wrong_year = (parsed_vn.dt.year != year).sum()
        if n_wrong_year > 0:
            raise ValueError(
                f"Có {n_wrong_year} bài viết ngoài năm {year}"
            )
    if month is not None:
        n_wrong_month = (parsed_vn.dt.month != month).sum()
        if n_wrong_month > 0:
            raise ValueError(
                f"Có {n_wrong_month} bài viết ngoài tháng {month:02d}/{year if year else '?'}"
            )
    df_copy[date_col] = parsed
    return df_copy


def check_categorical_domain(
    df: pd.DataFrame,
    valid_sources: Set[str] = VALID_SOURCES,
    valid_cats: Set[str] = VALID_CATS,
) -> None:
    
    actual_sources = set(df["source"].unique())
    if not actual_sources.issubset(valid_sources):
        raise ValueError(
            f"Nguồn báo không hợp lệ: {sorted(actual_sources - valid_sources)}"
        )

    actual_cats = set(df["category_gold"].unique())
    if not actual_cats.issubset(valid_cats):
        raise ValueError(
            f"Chuyên mục không hợp lệ: {sorted(actual_cats - valid_cats)}"
        )


def check_row_count(df: pd.DataFrame, expected_rows: Optional[int] = None) -> None:
    if expected_rows is None:
        logger.info("Bỏ qua kiểm tra số dòng (expected_rows=None)")
        return

    if len(df) != expected_rows:
        raise ValueError(
            f"Số dòng không khớp! Kỳ vọng: {expected_rows}, Thực tế: {len(df)}"
        )


def validate_raw_data(
    raw_path: Path = DEFAULT_RAW_CSV,
    output_path: Path = DEFAULT_OUTPUT_CSV,
    expected_rows: Optional[int] = None,
    target_year: Optional[int] = None,
    target_month: Optional[int] = None,
) -> pd.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {raw_path}")

    logger.info("Đọc file: %s", raw_path)
    df = pd.read_csv(
        raw_path,
        dtype={c: str for c in [
            "article_id", "url", "title", "content", "source", "category_gold",
        ]},
        low_memory=False,
    )
    logger.info("Đã đọc %d bài viết", len(df))

    logger.info("1/6 — Kiểm tra schema...")
    check_schema(df, REQUIRED_COLS)

    logger.info("2/6 — Kiểm tra article_id...")
    check_id_uniqueness(df, "article_id")

    logger.info("3/6 — Kiểm tra missing values...")
    check_missing_values(df, ["title", "content", "source", "url", "category_gold"])

    logger.info("4/6 — Kiểm tra định dạng ngày đăng...")
    df = check_datetime_range(df, "published_at", year=target_year, month=target_month)

    logger.info("5/6 — Kiểm tra source/category domain...")
    check_categorical_domain(df, VALID_SOURCES, VALID_CATS)

    logger.info("6/6 — Kiểm tra số dòng...")
    check_row_count(df, expected_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    try:
        display_path = output_path.relative_to(ROOT)
    except ValueError:
        display_path = output_path

    logger.info("✓ Thẩm định thành công. Đã lưu: %s", display_path)
    return df


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate dữ liệu thô cho pipeline NLP (Task D01)"
    )
    p.add_argument("--raw", type=Path, default=DEFAULT_RAW_CSV,
                   help=f"File CSV input (mặc định: {DEFAULT_RAW_CSV.name})")
    p.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_CSV,
                   help=f"File CSV output (mặc định: {DEFAULT_OUTPUT_CSV.name})")
    p.add_argument("--expected-rows", type=int, default=None,
                   help="Số dòng kỳ vọng. Bỏ trống để không check.")
    p.add_argument("--year", type=int, default=None,
                   help="Năm kỳ vọng. Bỏ trống để không check.")
    p.add_argument("--month", type=int, default=None,
                   help="Tháng kỳ vọng. Bỏ trống để không check.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        validate_raw_data(
            raw_path=args.raw,
            output_path=args.out,
            expected_rows=args.expected_rows,
            target_year=args.year,
            target_month=args.month,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error("Thẩm định thất bại: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()