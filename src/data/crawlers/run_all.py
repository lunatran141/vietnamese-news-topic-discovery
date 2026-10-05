"""Điều phối và hợp nhất dữ liệu từ VnExpress, Tuổi Trẻ, VietnamNet.

Hỗ trợ 2 chế độ vận hành:
    1. Chạy full pipeline (cào mới từ 3 báo → tự động gộp).
    2. Chỉ gộp (--merge-only) từ các file CSV đã cào sẵn trong data/raw/.

Cách dùng: Mở terminal trong thư mục của project
    # Chỉ gộp 3 file đã cào
    python -m src.data.crawlers.run_all --merge-only

    # Full pipeline với default date (2026-09-01 → 2026-09-30)
    python -m src.data.crawlers.run_all

    # Full pipeline với date cụ thể
    python -m src.data.crawlers.run_all --from-date 2026-09-01 --to-date 2026-09-30

    # Chỉ cào VietnamNet (debug)
    python -m src.data.crawlers.run_all --skip-vne --skip-ttr
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("master_runner")

# CONFIG
SCHEMA_COLUMNS: List[str] = [
    "article_id",
    "title",
    "content",
    "published_at",
    "source",
    "url",
    "category_gold",
]

RAW_DIR = Path("data/raw")
DEFAULT_FINAL_OUT = RAW_DIR / "news_articles.csv"

SOURCE_PATHS = {
    "vnexpress":  RAW_DIR / "vnexpress_articles.csv",
    "tuoitre":    RAW_DIR / "tuoitre_articles.csv",
    "vietnamnet": RAW_DIR / "vietnamnet_articles.csv",
}

SOURCE_DISPLAY_NAME = {
    "vnexpress":  "VnExpress",
    "tuoitre":    "Tuổi Trẻ",
    "vietnamnet": "VietnamNet",
}

# Default date (fallback khi user không truyền CLI arg)
DEFAULT_FROM_DATE: Tuple[int, int, int] = (2026, 9, 1)
DEFAULT_TO_DATE:   Tuple[int, int, int] = (2026, 9, 30)


# VALIDATION
def validate_and_clean_frame(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Kiểm tra schema và làm sạch cơ bản từng tập dữ liệu nguồn."""
    if df is None or df.empty:
        logger.warning("[%s] Tập dữ liệu rỗng.", source_name)
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
    if missing:
        logger.error("[%s] Thiếu các cột bắt buộc: %s", source_name, missing)
        raise ValueError(f"Dữ liệu từ {source_name} không đúng schema chuẩn.")

    clean_df = df[SCHEMA_COLUMNS].copy()

    before = len(clean_df)
    clean_df = clean_df.dropna(
        subset=["article_id", "title", "content", "category_gold"]
    )
    clean_df = clean_df[clean_df["content"].astype(str).str.strip().str.len() > 0]
    clean_df = clean_df[clean_df["title"].astype(str).str.strip().str.len() > 0]
    dropped = before - len(clean_df)

    if dropped > 0:
        logger.info("[%s] Đã loại bỏ %d bài viết lỗi/rỗng nội dung.",
                    source_name, dropped)

    return clean_df


def validate_post_merge(df: pd.DataFrame) -> None:
    """Kiểm tra tính toàn vẹn của corpus sau khi gộp."""
    if df is None or df.empty:
        return

    # 1. Không null ở cột định danh
    for col in ["article_id", "title", "content", "url", "category_gold"]:
        n_null = df[col].isna().sum()
        if n_null > 0:
            raise ValueError(f"Sau merge, cột '{col}' có {n_null} giá trị null")

    # 2. article_id không trùng
    n_dup = df["article_id"].duplicated().sum()
    if n_dup > 0:
        raise ValueError(f"Sau merge, phát hiện {n_dup} article_id trùng lặp")

    # 3. published_at parse được
    #    utc=True để tránh ValueError khi mix tz-aware và tz-naive (Pandas 2.0+)
    try:
        pd.to_datetime(df["published_at"], utc=True, errors="raise")
    except Exception as e:
        raise ValueError(f"published_at không parse được: {e}")


# MERGE
def _dedup_by_url_keep_longest(df: pd.DataFrame) -> pd.DataFrame:
    """Khi 2 bản ghi cùng URL, giữ bản có content dài hơn.
    Xử lý trường hợp 1 bài xuất hiện ở 2 chuyên mục khác nhau."""
    if df is None or df.empty:
        return df

    df = df.copy()
    df["_content_len"] = df["content"].astype(str).str.len()
    df = (
        df.sort_values("_content_len", ascending=False)
          .drop_duplicates(subset=["url"], keep="first")
          .drop(columns=["_content_len"])
          .reset_index(drop=True)
    )
    return df


def merge_datasets(
    dfs: List[Tuple[pd.DataFrame, str]],
    output_path: Path,
) -> pd.DataFrame:
    """Gộp các DataFrame, khử trùng lặp và lưu ra CSV."""
    cleaned_frames: List[pd.DataFrame] = []

    for df, name in dfs:
        # Defensive: crawler có thể trả None nếu có exception không mong đợi
        if df is not None and not df.empty:
            cleaned_frames.append(validate_and_clean_frame(df, name))
            logger.info("[%s] %d bài", name, len(df))
        else:
            logger.warning("[%s] Không có dữ liệu", name)

    if not cleaned_frames:
        logger.error("Không có dữ liệu hợp lệ nào để hợp nhất.")
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    merged = pd.concat(cleaned_frames, ignore_index=True)
    n_initial = len(merged)
    logger.info("Tổng ban đầu: %d bài", n_initial)

    # 1. Dedup URL — giữ content dài hơn
    merged = _dedup_by_url_keep_longest(merged)
    n_after_url = len(merged)

    # 2. Dedup article_id
    merged = merged.drop_duplicates(subset=["article_id"], keep="first")
    n_after_id = len(merged)

    if n_initial - n_after_url > 0:
        logger.info("  • Loại %d bài trùng URL", n_initial - n_after_url)
    if n_after_url - n_after_id > 0:
        logger.info("  • Loại %d bài trùng article_id", n_after_url - n_after_id)

    # 3. Sort theo thời gian + id để output deterministic
    merged = (
        merged.sort_values(by=["published_at", "article_id"])
              .reset_index(drop=True)
    )

    # 4. Validate tổng thể sau merge
    validate_post_merge(merged)

    # 5. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Đã lưu %d bài → %s", len(merged), output_path)

    return merged


# REPORT
def print_corpus_report(df: pd.DataFrame) -> None:
    """In báo cáo phân bố corpus phục vụ nghiệm thu dataset."""
    if df is None or df.empty:
        print("\n[!] Không có dữ liệu để lập báo cáo.")
        return

    print("\n" + "=" * 75)
    print("BÁO CÁO TẬP DỮ LIỆU ĐÃ HỢP NHẤT (NLP CORPUS SUMMARY)")
    print("=" * 75)

    print(f"Tổng số bài viết:  {len(df):,}")
    print(f"RAM:               {df.memory_usage().sum() / 1024 / 1024:.2f} MB")
    print(f"Khoảng thời gian:  {df['published_at'].min()}  -->  {df['published_at'].max()}")

    # Defensive astype(str) — tránh lỗi nếu có NaN lẫn trong content
    avg_words = df["content"].astype(str).str.split().str.len().mean()
    print(f"Độ dài TB:         {avg_words:.0f} từ/bài")

    print("\nMA TRẬN (CATEGORY × SOURCE):")
    cross_tab = pd.crosstab(
        df["category_gold"],
        df["source"],
        margins=True,
        margins_name="TỔNG",
    )
    print(cross_tab.to_string())

    print("\nKIỂM TRA CHẤT LƯỢNG:")
    print(f"  • article_id trùng:  {df['article_id'].duplicated().sum()}")
    print(f"  • URL trùng:         {df['url'].duplicated().sum()}")
    print(f"  • title null:        {df['title'].isna().sum()}")
    print(f"  • content null:      {df['content'].isna().sum()}")
    print("=" * 75 + "\n")


# PIPELINE
def _parse_date(val: str) -> Tuple[int, int, int]:
    dt = datetime.strptime(val, "%Y-%m-%d")
    return (dt.year, dt.month, dt.day)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Master Data Orchestrator cho 3 nguồn báo"
    )

    # default=None để tránh bug argparse không apply type cho default
    parser.add_argument("--from-date", type=_parse_date, default=None,
                        help="YYYY-MM-DD. Mặc định 2026-09-01")
    parser.add_argument("--to-date", type=_parse_date, default=None,
                        help="YYYY-MM-DD. Mặc định 2026-09-30")

    parser.add_argument("--max-pages-vne", type=int, default=25,
                        help="Số trang tối đa mỗi chuyên mục VnExpress")
    parser.add_argument("--max-pages-ttr", type=int, default=40,
                        help="Số trang timeline tối đa mỗi chuyên mục Tuổi Trẻ")
    parser.add_argument("--max-pages-vnn", type=int, default=18,
                        help="Số trang tối đa mỗi chuyên mục VietnamNet")

    parser.add_argument("--out", default=str(DEFAULT_FINAL_OUT),
                        help="Đường dẫn file output tổng hợp")

    parser.add_argument("--merge-only", action="store_true",
                        help="Chỉ gộp 3 file CSV đã cào, không gọi crawler")
    parser.add_argument("--skip-vne", action="store_true",
                        help="Bỏ qua VnExpress")
    parser.add_argument("--skip-ttr", action="store_true",
                        help="Bỏ qua Tuổi Trẻ")
    parser.add_argument("--skip-vnn", action="store_true",
                        help="Bỏ qua VietnamNet")

    args = parser.parse_args()

    # Xử lý default thủ công — vì argparse không apply type cho default
    from_date = args.from_date or DEFAULT_FROM_DATE
    to_date   = args.to_date   or DEFAULT_TO_DATE

    start_time = time.time()
    out_file = Path(args.out)
    data_sources: List[Tuple[pd.DataFrame, str]] = []

    # MERGE ONLY
    if args.merge_only:
        logger.info("Chế độ --merge-only: đọc CSV có sẵn trong %s", RAW_DIR)

        for key, path in SOURCE_PATHS.items():
            if args.skip_vne and key == "vnexpress":
                continue
            if args.skip_ttr and key == "tuoitre":
                continue
            if args.skip_vnn and key == "vietnamnet":
                continue

            if path.exists():
                logger.info("Đang đọc %s...", path)
                df = pd.read_csv(path, encoding="utf-8-sig")
                data_sources.append((df, SOURCE_DISPLAY_NAME[key]))
            else:
                logger.warning("Không tìm thấy: %s (bỏ qua)", path)

    # FULL PIPELINE
    else:
        logger.info("=== BẮT ĐẦU CÀO 3 TRANG BÁO %s → %s ===", from_date, to_date)

        # 1. VnExpress
        if not args.skip_vne:
            try:
                from src.data.crawlers.vnexpress import crawl_vnexpress
                logger.info(">>> [1/3] VnExpress...")
                vne_df = crawl_vnexpress(
                    from_date=from_date,
                    to_date=to_date,
                    max_pages_per_cat=args.max_pages_vne,
                    out_path=SOURCE_PATHS["vnexpress"],
                )
                data_sources.append((vne_df, "VnExpress"))
            except Exception as e:
                logger.error("Lỗi khi cào VnExpress: %s", e)

        # 2. Tuổi Trẻ
        if not args.skip_ttr:
            try:
                from src.data.crawlers.tuoitre import crawl_tuoitre
                logger.info(">>> [2/3] Tuổi Trẻ...")
                ttr_df = crawl_tuoitre(
                    from_date=from_date,
                    to_date=to_date,
                    max_pages_per_cat=args.max_pages_ttr,
                    out_path=SOURCE_PATHS["tuoitre"],
                )
                data_sources.append((ttr_df, "Tuổi Trẻ"))
            except Exception as e:
                logger.error("Lỗi khi cào Tuổi Trẻ: %s", e)

        # 3. VietnamNet
        if not args.skip_vnn:
            try:
                from src.data.crawlers.vietnamnet import crawl_vietnamnet
                logger.info(">>> [3/3] VietnamNet...")
                vnn_df = crawl_vietnamnet(
                    from_date=from_date,
                    to_date=to_date,
                    max_pages_per_cat=args.max_pages_vnn,
                    out_path=SOURCE_PATHS["vietnamnet"],
                )
                data_sources.append((vnn_df, "VietnamNet"))
            except Exception as e:
                logger.error("Lỗi khi cào VietnamNet: %s", e)

    # MERGE + SAVE 
    final_dataset = merge_datasets(data_sources, out_file)
    elapsed = time.time() - start_time

    print_corpus_report(final_dataset)
    logger.info("Hoàn tất trong %.2f phút.", elapsed / 60)


if __name__ == "__main__":
    main()