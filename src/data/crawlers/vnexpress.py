"""VnExpress crawler

Đặc điểm:
    - 7 chuyên mục, CateID đã xác thực.
    - Rate-limit + retry, an toàn với Wi-Fi/NAT dùng chung.
    - Drop diagnostics: KHÔNG nuốt lỗi.
    - published_at chuẩn ISO 8601 Asia/Ho_Chi_Minh.
    - Validate bài nằm trong [from_date 00:00, to_date 23:59:59].
    - article_id = SHA1(url không query) — ổn định.
    - Output sort deterministic theo article_id.

Cách dùng: Mở terminal trong thư mục của project
    # Smoke test 1 tuần
    python -m src.data.crawlers.vnexpress --from-date 2026-09-01 --to-date 2026-09-07  --out data/raw/_vne_smoke.csv

    # Full 1 tháng
    python -m src.data.crawlers.vnexpress --from-date 2026-09-01 --to-date 2026-09-30 --out data/raw/vnexpress_articles.csv
"""
from __future__ import annotations

import argparse
import hashlib
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
CATE_IDS: Dict[str, int] = {
    "Kinh doanh":          1003159,
    "Khoa học công nghệ":  1006219,
    "Thể thao":            1002565,
    "Giáo dục":            1003497,
    "Sức khỏe":            1003750,
    "Giải trí":            1002691,
    "Pháp luật":           1001007,
}

MIN_WORD_COUNT  = 50
MAX_WORKERS     = 6
REQUEST_TIMEOUT = 12
DELAY_SECONDS   = 0.1
TO_DATE_SLACK   = 3600   # bù timezone cho mép trên của to_date

HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br"
}

VN_TZ = timezone(timedelta(hours=7))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vnexpress")

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
    return "VNE_" + hashlib.sha1(clean.encode("utf-8")).hexdigest()[:12]


_DATE_RE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})[,\s|]+(\d{1,2}):(\d{2})")


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
        pass

    m = _DATE_RE.search(raw)
    if m:
        d, mo, y, h, mi = map(int, m.groups())
        try:
            return datetime(y, mo, d, h, mi, tzinfo=VN_TZ).isoformat()
        except ValueError:
            return ""
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


# STEP 1 — QUÉT URL TỪ TRANG PHÂN MỤC
def get_urls_from_page(session: requests.Session, page_url: str) -> List[str]:
    try:
        resp = session.get(page_url, timeout=REQUEST_TIMEOUT)
    except requests.Timeout:
        _inc_drop("list_timeout")
        return []
    except requests.RequestException as e:
        _inc_drop(f"list_error_{type(e).__name__}")
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
    for item in soup.find_all(["article", "div"], class_="item-news"):
        title_tag = item.find("h3", class_="title-news")
        if not title_tag:
            continue
        a_tag = title_tag.find("a")
        if not a_tag or not a_tag.get("href"):
            continue

        link = a_tag["href"]
        if link.startswith("/"):
            link = "https://vnexpress.net" + link

        if not link.startswith("http") or not link.endswith(".html"):
            continue
        if any(x in link for x in ("/video/", "/podcast/", "/anh/", "/interactive/")):
            continue

        urls.append(link)
    return urls


# STEP 2 — PARSE 1 BÀI
def parse_article_detail(
    session: requests.Session,
    url: str,
    category: str,
    from_ts: int,
    to_ts: int,
) -> Optional[Article]:
    # Fetch 
    try:
        time.sleep(DELAY_SECONDS)
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
    except requests.Timeout:
        _inc_drop("detail_timeout")
        return None
    except requests.RequestException as e:
        _inc_drop(f"detail_error_{type(e).__name__}")
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
    title_el = soup.find("h1", class_="title-detail") or soup.find("h1")
    if not title_el:
        _inc_drop("no_title")
        return None
    title = normalize_text(title_el.get_text())
    if not title:
        _inc_drop("empty_title")
        return None

    # Published at
    raw_date = ""
    meta = (
        soup.find("meta", {"name": "pubdate"})
        or soup.find("meta", {"property": "article:published_time"})
        or soup.find("meta", {"itemprop": "datePublished"})
    )
    if meta and meta.get("content"):
        raw_date = meta["content"].strip()
    else:
        d_el = soup.find("span", class_="date")
        if d_el:
            raw_date = d_el.get_text().strip()

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

    # Sapo (mô tả ngắn, thường nằm ngoài content_box) 
    sapo_el = soup.find("p", class_="description")
    sapo_text = normalize_text(sapo_el.get_text()) if sapo_el else ""

    # Content
    content_box = (
        soup.find("article", class_="fck_detail")
        or soup.find("div", class_="fck_detail")
        or soup.find("div", class_="Normal")
    )
    if not content_box:
        _inc_drop("no_content_box")
        return None

    paragraphs: List[str] = []
    seen_paragraphs: Set[str] = set()

    if sapo_text:
        paragraphs.append(sapo_text)
        seen_paragraphs.add(sapo_text)

    for p in content_box.find_all("p"):
        if p.find_parent("figure"):
            continue
        cls = p.get("class") or []
        if any(c in cls for c in ("Image", "Caption", "author_mail", "btn_send_comment")):
            continue
        txt = p.get_text().strip()
        if txt and txt not in seen_paragraphs:
            seen_paragraphs.add(txt)
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
        source="vnexpress",
        url=url,
        category_gold=category,
    )


# ORCHESTRATION
def crawl_vnexpress(
    from_date: Tuple[int, int, int],
    to_date: Tuple[int, int, int],
    max_pages_per_cat: int = 30,
    out_path: Path = Path("data/raw/vnexpress_articles.csv"),
) -> pd.DataFrame:
    from_ts = date_to_unix_ts(*from_date, is_end_of_day=False)
    to_ts   = date_to_unix_ts(*to_date,   is_end_of_day=True)

    session = create_session()

    # STEP 1: Quét URL từ trang phân mục 
    logger.info("=== STEP 1: Quét danh mục %s → %s ===", from_date, to_date)
    tasks: List[Tuple[str, str]] = []
    for cat_name, cate_id in CATE_IDS.items():
        for page in range(1, max_pages_per_cat + 1):
            tasks.append((
                cat_name,
                f"https://vnexpress.net/category/day/cateid/{cate_id}"
                f"/fromdate/{from_ts}/todate/{to_ts}"
                f"/allcate/{cate_id}/page/{page}",
            ))

    raw_urls_map: List[Tuple[str, str]] = []
    seen_urls: Set[str] = set()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_to_cat = {
            ex.submit(get_urls_from_page, session, url): cat
            for cat, url in tasks
        }
        for fut in tqdm(as_completed(future_to_cat), total=len(tasks), desc="Scan"):
            cat = future_to_cat[fut]
            for u in fut.result():
                if u not in seen_urls:
                    seen_urls.add(u)
                    raw_urls_map.append((u, cat))

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
    p = argparse.ArgumentParser(description="VnExpress production crawler")
    p.add_argument("--from-date", required=True, type=_parse_date,
                   help="YYYY-MM-DD (bao gồm)")
    p.add_argument("--to-date", required=True, type=_parse_date,
                   help="YYYY-MM-DD (bao gồm)")
    p.add_argument("--max-pages", type=int, default=40,
                   help="Số trang tối đa mỗi chuyên mục (mặc định 40, đủ cho ~1 tháng)")
    p.add_argument("--out", default="data/raw/vnexpress_articles.csv",
                   help="Đường dẫn CSV output")
    args = p.parse_args()

    t0 = time.time()
    df = crawl_vnexpress(
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