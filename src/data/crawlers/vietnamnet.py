"""VietnamNet production crawler.

Đặc điểm:
    - 7 chuyên mục, slug đã xác thực.
    - 6 workers, rate-limit an toàn.
    - Drop diagnostics đầy đủ.
    - Sapo có class filter (không bắt nhầm h2 menu).
    - published_at chuẩn ISO 8601 Asia/Ho_Chi_Minh (ưu tiên JSON-LD).
    - article_id = SHA1(url không query) — ổn định.
    - Output sort deterministic theo article_id.

Cách dùng: Mở terminal trong thư mục của project
    # Smoke test 1 tuần
    python -m src.data.crawlers.vietnamnet --from-date 2026-09-01 --to-date 2026-09-07 --out data/raw/_vnn_smoke.csv

    # Full 1 tháng
    python -m src.data.crawlers.vietnamnet --from-date 2026-09-01 --to-date 2026-09-30  --out data/raw/vietnamnet_articles.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util import Retry


# CONFIG
VIETNAMNET_SLUGS: Dict[str, str] = {
    "Kinh doanh":          "kinh-doanh",
    "Khoa học công nghệ":  "cong-nghe",
    "Thể thao":            "the-thao",
    "Giáo dục":            "giao-duc",
    "Sức khỏe":            "suc-khoe",
    "Pháp luật":           "thoi-su/phap-luat",
    "Giải trí":            "van-hoa-giai-tri",
}

MIN_WORD_COUNT  = 50
MAX_WORKERS     = 6
REQUEST_TIMEOUT = 12
DELAY_ARTICLE   = 0.1
DELAY_PAGE      = 0.3
TO_DATE_SLACK   = 3600   # 1 giờ, bù timezone cho mép trên

HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

VN_TZ = timezone(timedelta(hours=7))

# VietnamNet dùng ID số ở cuối URL
_URL_ID_RE = re.compile(r"-\d{6,}\.html?$")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vietnamnet")

# DROP DIAGNOSTICS — thread-safe
DROP_STATS: Counter = Counter()
_drop_lock = threading.Lock()


def _inc_drop(key: str) -> None:
    with _drop_lock:
        DROP_STATS[key] += 1


# SCHEMA
@dataclass
class Article:
    article_id:    str
    title:         str
    content:       str
    published_at:  str
    source:        str
    url:           str
    category_gold: str


# TIME UTILITIES
def date_to_unix_ts(y: int, m: int, d: int, is_end_of_day: bool = False) -> int:
    h, mi, s = (23, 59, 59) if is_end_of_day else (0, 0, 0)
    return int(datetime(y, m, d, h, mi, s, tzinfo=VN_TZ).timestamp())


# TEXT UTILITIES
def make_article_id(url: str) -> str:
    clean = urlparse(url)._replace(query="", fragment="").geturl()
    return "VNN_" + hashlib.sha1(clean.encode("utf-8")).hexdigest()[:12]


def normalize_published_at(raw: str) -> str:
    if not raw:
        return ""
    raw = raw.strip()
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=VN_TZ)
        return dt.astimezone(VN_TZ).isoformat()
    except ValueError:
        return ""


def normalize_text(raw: str) -> str:
    return " ".join(raw.split()).strip() if raw else ""


# NETWORK
def create_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(
        max_retries=retries,
        pool_connections=MAX_WORKERS,
        pool_maxsize=MAX_WORKERS,
    )
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


# STEP 1 — QUÉT URL TỪ TRANG DANH MỤC
def get_urls_from_page(
    session: requests.Session,
    slug: str,
    page: int,
) -> List[str]:
    """Trả về danh sách URL bài viết từ 1 trang danh mục."""
    url = f"https://vietnamnet.vn/{slug}" if page == 1 \
          else f"https://vietnamnet.vn/{slug}-page{page}"

    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
    except requests.Timeout:
        _inc_drop("list_timeout")
        return []
    except requests.RequestException as e:
        _inc_drop(f"list_err_{type(e).__name__}")
        return []

    if resp.status_code != 200:
        _inc_drop(f"list_http_{resp.status_code}")
        return []

    try:
        soup = BeautifulSoup(resp.content, "html.parser")
    except Exception as e:
        _inc_drop(f"list_parse_{type(e).__name__}")
        return []

    urls: List[str] = []
    seen_in_page: Set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not _URL_ID_RE.search(href):
            continue
        if any(x in href for x in ("/video/", "/podcast/", "/talkshow/", "/infographic/")):
            continue

        full = "https://vietnamnet.vn" + href if href.startswith("/") else href
        if not full.startswith("https://vietnamnet.vn/"):
            continue
        if full in seen_in_page:
            continue
        seen_in_page.add(full)
        urls.append(full)

    return urls


# STEP 2 — PARSE 1 BÀI
def parse_article_detail(
    session: requests.Session,
    url: str,
    category: str,
    from_ts: int,
    to_ts: int,
) -> Optional[Article]:
    # ---- Fetch ----
    try:
        time.sleep(DELAY_ARTICLE)
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
    except requests.Timeout:
        _inc_drop("detail_timeout")
        return None
    except requests.RequestException as e:
        _inc_drop(f"detail_err_{type(e).__name__}")
        return None

    if resp.status_code != 200:
        _inc_drop(f"detail_http_{resp.status_code}")
        return None

    try:
        soup = BeautifulSoup(resp.content, "html.parser")
    except Exception as e:
        _inc_drop(f"detail_parse_{type(e).__name__}")
        return None

    # Title 
    title_el = (
        soup.find("h1", class_="content-detail-title")
        or soup.find("h1", class_="article-detail-title")
        or soup.find("h1")
    )
    if not title_el:
        _inc_drop("no_title")
        return None
    title = normalize_text(title_el.get_text())
    if not title:
        _inc_drop("empty_title")
        return None

    # Published at (ưu tiên JSON-LD) 
    raw_date = ""
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(s.string or "{}")
            if isinstance(data, dict) and "datePublished" in data:
                raw_date = data["datePublished"]
                break
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "datePublished" in item:
                        raw_date = item["datePublished"]
                        break
                if raw_date:
                    break
        except Exception:
            continue

    if not raw_date:
        meta = (
            soup.find("meta", {"property": "article:published_time"})
            or soup.find("meta", {"name": "pubdate"})
            or soup.find("meta", {"itemprop": "datePublished"})
        )
        if meta and meta.get("content"):
            raw_date = meta["content"].strip()

    published_at = normalize_published_at(raw_date)
    if not published_at:
        _inc_drop("bad_date")
        return None

    try:
        pub_dt = datetime.fromisoformat(published_at)
        if pub_dt.tzinfo is None:
            pub_dt = pub_dt.replace(tzinfo=VN_TZ)
        pub_ts = pub_dt.timestamp()
    except ValueError:
        _inc_drop("bad_date_parse")
        return None

    if not (from_ts <= pub_ts <= to_ts + TO_DATE_SLACK):
        _inc_drop("out_of_range")
        return None

    # Sapo 
    sapo_el = (
        soup.find("h2", class_="content-detail-sapo")
        or soup.find("div", class_="content-detail-sapo")
    )
    sapo_text = normalize_text(sapo_el.get_text()) if sapo_el else ""

    # Content 
    content_box = (
        soup.find("div", class_="maincontent")
        or soup.find("div", id="maincontent")
        or soup.find("div", class_="content-detail")
    )
    if not content_box:
        _inc_drop("no_content_box")
        return None

    paragraphs: List[str] = []
    seen: Set[str] = set()

    if sapo_text:
        paragraphs.append(sapo_text)
        seen.add(sapo_text)

    for p in content_box.find_all("p"):
        if p.find_parent("figure") or p.find_parent("table"):
            continue
        p_cls = p.get("class") or []
        if any(c in p_cls for c in ("Image", "Caption", "author-info")):
            continue
        txt = normalize_text(p.get_text())
        if txt and txt not in seen:
            seen.add(txt)
            paragraphs.append(txt)

    content = normalize_text(" ".join(paragraphs))
    if len(content.split()) < MIN_WORD_COUNT:
        _inc_drop("too_short")
        return None

    return Article(
        article_id=make_article_id(url),
        title=title,
        content=content,
        published_at=published_at,
        source="vietnamnet",
        url=url,
        category_gold=category,
    )


# ORCHESTRATION
def crawl_vietnamnet(
    from_date: Tuple[int, int, int],
    to_date: Tuple[int, int, int],
    max_pages_per_cat: int = 18,
    out_path: Path = Path("data/raw/vietnamnet_articles.csv"),
) -> pd.DataFrame:
    from_ts = date_to_unix_ts(*from_date, is_end_of_day=False)
    to_ts   = date_to_unix_ts(*to_date,   is_end_of_day=True)

    session = create_session()

    # STEP 1: Quét URL từ trang danh mục
    logger.info("=== STEP 1: Quét danh mục %s → %s ===", from_date, to_date)
    raw_urls_map: List[Tuple[str, str]] = []
    seen_urls: Set[str] = set()

    for cat_idx, (cat_name, slug) in enumerate(VIETNAMNET_SLUGS.items(), start=1):
        logger.info("[%d/%d] %s (slug=%s)", cat_idx, len(VIETNAMNET_SLUGS), cat_name, slug)
        cat_total = 0

        for page in tqdm(range(1, max_pages_per_cat + 1),
                         desc=f"  {cat_name}", leave=False):
            urls = get_urls_from_page(session, slug, page)
            if not urls:
                break
            for u in urls:
                if u not in seen_urls:
                    seen_urls.add(u)
                    raw_urls_map.append((u, cat_name))
                    cat_total += 1
            time.sleep(DELAY_PAGE)

        logger.info("    → %s: %d URL", cat_name, cat_total)

    logger.info("Tìm thấy %d URL duy nhất", len(raw_urls_map))

    # STEP 2: Parse chi tiết 
    logger.info("=== STEP 2: Parse chi tiết (%d luồng) ===", MAX_WORKERS)
    collected: List[dict] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_to_url = {
            ex.submit(parse_article_detail, session, url, cat, from_ts, to_ts): url
            for url, cat in raw_urls_map
        }
        for fut in tqdm(as_completed(future_to_url),
                        total=len(raw_urls_map), desc="Parse"):
            art = fut.result()
            if art is not None:
                collected.append(asdict(art))

    # STEP 3: Lưu output 
    df = pd.DataFrame(collected)
    if not df.empty:
        df = (
            df.drop_duplicates(subset=["article_id"])
              .sort_values("article_id")
              .reset_index(drop=True)
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Đã lưu %d bài → %s", len(df), out_path)

    return df


# REPORTING
def print_diagnostics(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("DROP DIAGNOSTICS")
    print("=" * 60)
    if not DROP_STATS:
        print("  (không có bài nào bị loại)")
    for k, v in DROP_STATS.most_common():
        print(f"  {k:30s} {v:>8d}")

    print("\n" + "=" * 60)
    print("PHÂN BỐ THEO CHUYÊN MỤC")
    print("=" * 60)
    if not df.empty:
        print(df["category_gold"].value_counts().to_string())
        print(f"\n  Tổng: {len(df)} bài")
    else:
        print("  (rỗng)")


# CLI
def _parse_date(s: str) -> Tuple[int, int, int]:
    dt = datetime.strptime(s, "%Y-%m-%d")
    return (dt.year, dt.month, dt.day)


def main() -> None:
    p = argparse.ArgumentParser(description="VietnamNet production crawler")
    p.add_argument("--from-date", required=True, type=_parse_date,
                   help="YYYY-MM-DD (bao gồm)")
    p.add_argument("--to-date", required=True, type=_parse_date,
                   help="YYYY-MM-DD (bao gồm)")
    p.add_argument("--max-pages", type=int, default=30,
                   help="Số trang tối đa mỗi chuyên mục (mặc định 30)")
    p.add_argument("--out", default="data/raw/vietnamnet_articles.csv",
                   help="Đường dẫn CSV output")
    args = p.parse_args()

    t0 = time.time()
    df = crawl_vietnamnet(
        from_date=args.from_date,
        to_date=args.to_date,
        max_pages_per_cat=args.max_pages,
        out_path=Path(args.out),
    )
    elapsed = time.time() - t0

    print_diagnostics(df)
    print(f"\nThời gian chạy: {elapsed:.1f}s ({elapsed / 60:.2f} phút)")


if __name__ == "__main__":
    main()