"""Thu thập và khóa dữ liệu E07 từ ngày 01–15/10/2026."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

from src.data.crawlers.run_all import SCHEMA_COLUMNS, merge_datasets
from src.data.crawlers.tuoitre import crawl_tuoitre
from src.data.crawlers.vietnamnet import crawl_vietnamnet
from src.data.crawlers.vnexpress import crawl_vnexpress
from src.preprocessing.clean_text import clean_text_p0


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data/ood/oct2026"
SEPTEMBER_PATH = ROOT / "data/raw/news_articles.csv"

FROM_DATE = (2026, 10, 1)
TO_DATE = (2026, 10, 15)

# Đã chốt trong docs/e07_dataset_plan.md.
# Mỗi tuple: tên nguồn, hàm crawler, số trang tối đa mỗi chuyên mục.
SOURCES = (
    ("vnexpress", crawl_vnexpress, 25),
    ("tuoitre", crawl_tuoitre, 40),
    ("vietnamnet", crawl_vietnamnet, 18),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def normalize_url(value: str) -> str:
    """Bỏ query và dấu / cuối URL để so trùng."""
    parsed = urlsplit(str(value).strip())
    return f"{parsed.netloc.lower()}{parsed.path.rstrip('/')}"


def normalize_text(value: str) -> str:
    """Chuẩn hóa nhẹ để so bài trùng chính xác theo title/content."""
    text = unicodedata.normalize("NFC", str(value))
    return re.sub(r"\s+", " ", text).strip().casefold()


def check_required_values(
    frame: pd.DataFrame,
    label: str,
) -> None:
    missing = set(SCHEMA_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} thiếu cột: {sorted(missing)}")

    if frame.empty:
        raise ValueError(
            f"{label} không có bài. Dừng để kiểm tra crawler."
        )

    required = [
        "article_id",
        "title",
        "content",
        "published_at",
        "source",
        "url",
        "category_gold",
    ]

    if frame[required].isna().any().any():
        raise ValueError(f"{label} có giá trị bắt buộc bị thiếu.")

    for column in required:
        if frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{label} có {column} rỗng.")


def check_dates(
    frame: pd.DataFrame,
    label: str,
) -> None:
    published = pd.to_datetime(
        frame["published_at"],
        errors="raise",
        utc=True,
    )

    dates_vn = (
        published
        .dt.tz_convert("Asia/Ho_Chi_Minh")
        .dt.date
    )

    first_day = datetime(*FROM_DATE).date()
    last_day = datetime(*TO_DATE).date()
    invalid = ~dates_vn.between(first_day, last_day)

    if invalid.any():
        examples = frame.loc[
            invalid,
            ["article_id", "published_at"],
        ].head(3)

        raise ValueError(
            f"{label} có {int(invalid.sum())} bài ngoài "
            "khoảng 01–15/10/2026:\n"
            f"{examples.to_string(index=False)}"
        )


def exclude_september_overlaps(
    october: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loại bài trùng tháng 9 và trả thêm bảng ghi lý do loại."""
    if not SEPTEMBER_PATH.is_file():
        raise FileNotFoundError(
            f"Thiếu bộ dữ liệu tháng 9: {SEPTEMBER_PATH}"
        )

    september = pd.read_csv(
        SEPTEMBER_PATH,
        encoding="utf-8-sig",
        dtype={"article_id": str, "url": str},
    )
    check_required_values(september, "Bộ tháng 9")

    september_ids = set(september["article_id"])
    september_urls = set(
        september["url"].map(normalize_url)
    )
    september_texts = set(
        zip(
            september["title"].map(normalize_text),
            september["content"].map(normalize_text),
        )
    )

    id_hits = october["article_id"].isin(september_ids)

    url_hits = (
        october["url"]
        .map(normalize_url)
        .isin(september_urls)
    )

    text_hits = pd.Series(
        [
            (
                normalize_text(title),
                normalize_text(content),
            ) in september_texts
            for title, content in zip(
                october["title"],
                october["content"],
            )
        ],
        index=october.index,
    )

    overlap = id_hits | url_hits | text_hits

    excluded = october.loc[overlap].copy()
    excluded["overlap_reason"] = [
        ",".join(
            reason
            for reason, hit in (
                ("article_id", id_hit),
                ("url", url_hit),
                ("title_content", text_hit),
            )
            if hit
        )
        for id_hit, url_hit, text_hit in zip(
            id_hits.loc[overlap],
            url_hits.loc[overlap],
            text_hits.loc[overlap],
        )
    ]

    kept = (
        october.loc[~overlap]
        .reset_index(drop=True)
    )

    if kept.empty:
        raise ValueError(
            "Không còn bài E07 sau khi loại bài trùng tháng 9."
        )

    print(
        f"Đã loại {len(excluded)} bài trùng tháng 9; "
        f"giữ {len(kept)} bài E07."
    )

    return kept, excluded


def main() -> None:
    # Thư mục có thể đã được tạo thủ công, nhưng phải rỗng.
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"{OUTPUT_DIR} đã có file. Không ghi đè "
            "bộ dữ liệu E07 đã tạo."
        )

    if not SEPTEMBER_PATH.is_file():
        raise FileNotFoundError(
            f"Thiếu bộ dữ liệu tháng 9: {SEPTEMBER_PATH}"
        )

    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)

    # Crawler ghi vào thư mục tạm. Chỉ xuất bộ E07 chính thức
    # sau khi đủ ba nguồn và tất cả kiểm tra đều đạt.
    with tempfile.TemporaryDirectory(
        prefix="e07_collect_",
        dir=OUTPUT_DIR.parent,
    ) as temporary:
        staging_dir = Path(temporary)
        frames = []
        source_counts_before_merge = {}

        for source_name, crawl_function, max_pages in SOURCES:
            source_path = (
                staging_dir
                / f"{source_name}_articles.csv"
            )

            print(f"Đang thu thập {source_name}...")

            frame = crawl_function(
                from_date=FROM_DATE,
                to_date=TO_DATE,
                max_pages_per_cat=max_pages,
                out_path=source_path,
            )

            if frame is None:
                raise ValueError(
                    f"{source_name} không trả về dữ liệu."
                )

            check_required_values(frame, source_name)
            check_dates(frame, source_name)

            actual_sources = set(
                frame["source"]
                .astype(str)
                .str.casefold()
            )
            if actual_sources != {source_name}:
                raise ValueError(
                    f"{source_name} có giá trị source "
                    f"không khớp: {actual_sources}"
                )

            if frame["article_id"].duplicated().any():
                raise ValueError(
                    f"{source_name} có article_id trùng."
                )

            source_counts_before_merge[source_name] = (
                len(frame)
            )
            frames.append((frame, source_name))

            print(
                f"{source_name}: {len(frame)} bài "
                "qua kiểm tra ban đầu"
            )

        merged_path = staging_dir / "news_articles.csv"
        merged = merge_datasets(frames, merged_path)

        check_required_values(merged, "Bộ tháng 10")
        check_dates(merged, "Bộ tháng 10")

        if merged["article_id"].duplicated().any():
            raise ValueError(
                "Bộ tháng 10 có article_id trùng."
            )

        normalized_urls = (
            merged["url"].map(normalize_url)
        )
        if normalized_urls.duplicated().any():
            raise ValueError(
                "Bộ tháng 10 có URL trùng."
            )

        expected_sources = {
            name for name, _, _ in SOURCES
        }
        actual_sources = set(
            merged["source"]
            .astype(str)
            .str.casefold()
        )
        if actual_sources != expected_sources:
            raise ValueError(
                "Bộ tháng 10 thiếu nguồn sau khi gộp: "
                f"{expected_sources - actual_sources}"
            )

        n_articles_before_september_filter = len(merged)

        merged, excluded = (
            exclude_september_overlaps(merged)
        )

        remaining_sources = set(
            merged["source"]
            .astype(str)
            .str.casefold()
        )
        if remaining_sources != expected_sources:
            raise ValueError(
                "Sau khi loại bài trùng, bộ E07 thiếu nguồn: "
                f"{expected_sources - remaining_sources}"
            )

        # merge_datasets đã ghi CSV trước khi loại bài trùng.
        # Ghi lại đúng bộ dữ liệu cuối cùng sẽ được đánh giá.
        merged.to_csv(
            merged_path,
            index=False,
            encoding="utf-8-sig",
        )

        excluded_path = (
            staging_dir
            / "excluded_september_overlap.csv"
        )
        excluded.to_csv(
            excluded_path,
            index=False,
            encoding="utf-8-sig",
        )

        # Dùng đúng bước P0 như embedding trên train.
        processed = merged.copy()
        processed["processed_title"] = (
            processed["title"].map(clean_text_p0)
        )
        processed["processed_content"] = (
            processed["content"].map(clean_text_p0)
        )

        if processed["processed_title"].eq("").any():
            raise ValueError(
                "Có processed_title rỗng sau P0."
            )
        if processed["processed_content"].eq("").any():
            raise ValueError(
                "Có processed_content rỗng sau P0."
            )

        processed_path = (
            staging_dir / "news_articles_p0.csv"
        )
        processed.to_csv(
            processed_path,
            index=False,
            encoding="utf-8-sig",
        )

        counts_by_date = (
            pd.to_datetime(
                merged["published_at"],
                utc=True,
            )
            .dt.tz_convert("Asia/Ho_Chi_Minh")
            .dt.strftime("%Y-%m-%d")
            .value_counts()
            .sort_index()
        )

        file_hashes = {
            "news_articles.csv": sha256_file(
                merged_path
            ),
            "news_articles_p0.csv": sha256_file(
                processed_path
            ),
            "excluded_september_overlap.csv": (
                sha256_file(excluded_path)
            ),
        }

        for source_name, _, _ in SOURCES:
            filename = f"{source_name}_articles.csv"
            file_hashes[filename] = sha256_file(
                staging_dir / filename
            )

        manifest = {
            "task": "E07",
            "dataset": "oct2026_01_15",
            "date_from": "2026-10-01",
            "date_to": "2026-10-15",
            "timezone": "Asia/Ho_Chi_Minh",
            "collected_at_utc": (
                datetime.now(timezone.utc).isoformat()
            ),
            "n_articles": len(merged),
            "n_articles_before_september_filter": (
                n_articles_before_september_filter
            ),
            "n_excluded_september_overlap": (
                len(excluded)
            ),
            "source_counts_before_merge": (
                source_counts_before_merge
            ),
            "source_counts_after_merge": {
                str(source): int(count)
                for source, count in (
                    merged["source"]
                    .value_counts()
                    .items()
                )
            },
            "counts_by_date": {
                str(day): int(count)
                for day, count in (
                    counts_by_date.items()
                )
            },
            "preprocessing": (
                "clean_text_p0 on title and content"
            ),
            "september_reference": (
                str(SEPTEMBER_PATH.relative_to(ROOT))
                .replace("\\", "/")
            ),
            "september_sha256": (
                sha256_file(SEPTEMBER_PATH)
            ),
            "max_pages_per_category": {
                source_name: max_pages
                for source_name, _, max_pages in SOURCES
            },
            "files": file_hashes,
        }

        # Chỉ chép ra thư mục E07 chính thức khi đã kiểm tra xong.
        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        for source_name, _, _ in SOURCES:
            filename = f"{source_name}_articles.csv"
            shutil.copy2(
                staging_dir / filename,
                OUTPUT_DIR / filename,
            )

        shutil.copy2(
            merged_path,
            OUTPUT_DIR / merged_path.name,
        )
        shutil.copy2(
            processed_path,
            OUTPUT_DIR / processed_path.name,
        )
        shutil.copy2(
            excluded_path,
            OUTPUT_DIR / excluded_path.name,
        )

        # Manifest được ghi cuối cùng.
        manifest_path = (
            OUTPUT_DIR / "manifest.json"
        )
        manifest_path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(
        f"Đã khóa {manifest['n_articles']} bài "
        f"trong: {OUTPUT_DIR}"
    )
    print(
        "Đã loại bài trùng tháng 9:",
        manifest["n_excluded_september_overlap"],
    )
    print(
        "Số bài theo nguồn:",
        manifest["source_counts_after_merge"],
    )
    print(
        "Số bài theo ngày:",
        manifest["counts_by_date"],
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()