"""Tuổi Trẻ production crawler

Đặc điểm:
    - 7 chuyên mục, Zone ID đã xác thực.
    - Session có Referer + X-Requested-With cho timeline endpoint.
    - Bóc tách Sapo (<h2 class="sapo"> / <h2 class="article-sapo">).
    - published_at chuẩn ISO 8601 Asia/Ho_Chi_Minh.
    - Validate bài nằm trong [from_date 00:00, to_date 23:59:59].
    - article_id = SHA1(url không query) — ổn định.
    - Output sort deterministic theo article_id.

Cách dùng: Mở terminal trong thư mục của project
    # Smoke test 1 tuần
    python -m src.data.crawlers.tuoitre --from-date 2026-09-01 --to-date 2026-09-07 --out data/raw/_tt_smoke.csv

    # Full 1 tháng
    python -m src.data.crawlers.tuoitre --from-date 2026-09-01 --to-date 2026-09-30 --out data/raw/tuoitre_articles.csv
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
TUOITRE_ZONES: Dict[str, Dict] = {
    "Kinh doanh":          {"zone_id": 11,     "referer": "https://tuoitre.vn/kinh-doanh.htm"},
    "Khoa học công nghệ":  {"zone_id": 200029, "referer": "https://tuoitre.vn/cong-nghe.htm"},
    "Thể thao":            {"zone_id": 1209,   "referer": "https://tuoitre.vn/the-thao.htm"},
    "Giáo dục":            {"zone_id": 13,     "referer": "https://tuoitre.vn/giao-duc.htm"},
    "Sức khỏe":            {"zone_id": 12,     "referer": "https://tuoitre.vn/suc-khoe.htm"},
    "Pháp luật":           {"zone_id": 6,      "referer": "https://tuoitre.vn/phap-luat.htm"},
    "Giải trí":            {"zone_id": 10,     "referer": "https://tuoitre.vn/giai-tri.htm"},
}

MIN_WORD_COUNT  = 50
MAX_WORKERS     = 6
REQUEST_TIMEOUT = 12
DELAY_SECONDS   = 0.08
TO_DATE_SLACK   = 3600   # bù timezone cho mép trên của to_date

HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
}

VN_TZ = timezone(timedelta(hours=7))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tuoitre")


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
    return "TTR_" + hashlib.sha1(clean.encode("utf-8")).hexdigest()[:12]


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


# STEP 1 — QUÉT TIMELINE PAGE
def get_urls_from_page(
    session: requests.Session,
    zone_id: int,
    referer: str,
    page: int,
) -> List[str]:
    timeline_url = f"https://tuoitre.vn/timeline/{zone_id}/trang-{page}.htm"
    headers = {
        "Referer": referer,
        "X-Requested-With": "XMLHttpRequest",
    }
    try:
        resp = session.get(timeline_url, headers=headers, timeout=REQUEST_TIMEOUT)
    except requests.Timeout:
        _inc_drop("list_timeout")
        return []
    except requests.RequestException as e:
        _inc_drop(f"list_error_{type(e).__name__}")
        return []

    if resp.status_code != 200:
        _inc_drop(f"list_http_{resp.status_code}")
        return []
    if not resp.text.strip():
        return []   # page rỗng — kết thúc zone bình thường

    try:
        soup = BeautifulSoup(resp.content, "html.parser")
    except Exception as e:
        _inc_drop(f"list_parse_{type(e).__name__}")
        return []

    urls: List[str] = []
    items = (
        soup.find_all("a", class_="box-category-link-title")
        or soup.find_all("a", class_="box-category-title-text")
    )
    for a_tag in items:
        href = a_tag.get("href", "")
        if not href or not href.endswith(".htm"):
            continue
        if any(x in href for x in ("/video/", "/podcast/", "/media/", "/infographic/")):
            continue
        full = "https://tuoitre.vn" + href if href.startswith("/") else href
        if full.startswith("http"):
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

    # ---- Title ----
    title_el = (
        soup.find("h1", class_="article-title")
        or soup.find("h1", class_="title-detail")
        or soup.find("h1")
    )
    if not title_el:
        _inc_drop("no_title")
        return None
    title = normalize_text(title_el.get_text())
    if not title:
        _inc_drop("empty_title")
        return None

    # Published at
    raw_date = ""
    meta = soup.find("meta", {"property": "article:published_time"})
    if meta and meta.get("content"):
        raw_date = meta["content"].strip()
    else:
        d_el = soup.find("div", class_="date-time") or soup.find("span", class_="date-time")
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

    # Sapo (tóm tắt đầu bài)
    # Tuổi Trẻ dùng <h2 class="sapo"> hoặc <h2 class="article-sapo">.
    # Sapo thường nằm ngoài detail-cmain, là <h2> chứ không phải <p>.
    sapo_el = (
        soup.find("h2", class_="sapo")
        or soup.find("h2", class_="article-sapo")
        or soup.find("p", class_="sapo")
        or soup.find("div", class_="sapo")
    )
    sapo_text = normalize_text(sapo_el.get_text()) if sapo_el else ""

    # ---- Content ----
    content_box = (
        soup.find("div", class_="detail-cmain")
        or soup.find("div", id="main-detail-body")
        or soup.find("div", class_="content fck")
    )
    if not content_box:
        _inc_drop("no_content_box")
        return None

    paragraphs: List[str] = []
    seen_paragraphs: Set[str] = set()

    # Sapo đi trước — mật độ từ khóa cô đọng nhất, giữ vị trí đầu để
    # TF-IDF / embedding ưu tiên đúng tín hiệu mở bài.
    if sapo_text:
        paragraphs.append(sapo_text)
        seen_paragraphs.add(sapo_text)

    for p in content_box.find_all(["p", "h2"]):
        # Bỏ caption ảnh
        if p.find_parent("figure"):
            continue

        # Bỏ chính sapo nếu nó nằm trong content_box (tránh trùng)
        p_cls = p.get("class") or []
        if "sapo" in p_cls or "article-sapo" in p_cls:
            continue

        # Bỏ paragraph rác
        if any(c in p_cls for c in ("VCSortableInPreviewMode", "Image", "Caption", "author_mail")):
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
        source="tuoitre",
        url=url,
        category_gold=category,
    )


# ORCHESTRATION
def crawl_tuoitre(
    from_date: Tuple[int, int, int],
    to_date: Tuple[int, int, int],
    max_pages_per_cat: int = 80,
    out_path: Path = Path("data/raw/tuoitre_articles.csv"),
) -> pd.DataFrame:
    from_ts = date_to_unix_ts(*from_date, is_end_of_day=False)
    to_ts   = date_to_unix_ts(*to_date,   is_end_of_day=True)

    session = create_session()

    # STEP 1: Quét timeline từng zone
    logger.info("=== STEP 1: Quét timeline %s → %s ===", from_date, to_date)
    raw_urls_map: List[Tuple[str, str]] = []
    seen_urls: Set[str] = set()

    for cat_name, info in TUOITRE_ZONES.items():
        zone_id = info["zone_id"]
        referer = info["referer"]
        logger.info("Quét zone %d (%s)", zone_id, cat_name)

        for page in tqdm(range(1, max_pages_per_cat + 1),
                         desc=f"  {cat_name}", leave=False):
            urls = get_urls_from_page(session, zone_id, referer, page)
            if not urls:
                break
            for u in urls:
                if u not in seen_urls:
                    seen_urls.add(u)
                    raw_urls_map.append((u, cat_name))
            time.sleep(DELAY_SECONDS)

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
    p = argparse.ArgumentParser(description="Tuổi Trẻ production crawler")
    p.add_argument("--from-date", required=True, type=_parse_date,
                   help="YYYY-MM-DD (bao gồm)")
    p.add_argument("--to-date", required=True, type=_parse_date,
                   help="YYYY-MM-DD (bao gồm)")
    p.add_argument("--max-pages", type=int, default=50,
                   help="Số trang timeline tối đa mỗi chuyên mục (mặc định 50)")
    p.add_argument("--out", default="data/raw/tuoitre_articles.csv",
                   help="Đường dẫn CSV output")
    args = p.parse_args()

    t0 = time.time()
    df = crawl_tuoitre(
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