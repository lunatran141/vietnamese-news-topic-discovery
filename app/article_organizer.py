"""Dịch vụ bóc tách, tiền xử lý, gán chủ đề và gom nhóm sự kiện cho bài báo mới."""

from __future__ import annotations

import hashlib
import io
import json
import re
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.data_service import ARTIFACT_ROOT, REPO_ROOT, ArtifactRun, enrich_documents
from src.artifacts.loader import ArtifactBundle
from src.preprocessing.clean_text import clean_text_p0
from src.preprocessing.normalize_text import normalize_text_p1
from src.preprocessing.remove_stopwords import load_stopwords, remove_stopwords_p3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
}

STOPWORDS_PATH = REPO_ROOT / "configs" / "vietnamese_stopwords.txt"
TFIDF_DIR = REPO_ROOT / "artifacts" / "tfidf"
KMEANS_MODEL_PATH = REPO_ROOT / "artifacts" / "kmeans" / "kmeans_model.joblib"
LDA_PIPELINE_PATH = REPO_ROOT / "artifacts" / "lda" / "final" / "lda_pipeline.joblib"
NMF_PIPELINE_PATH = (
    REPO_ROOT
    / "artifacts"
    / "nmf"
    / "nmf_p3_k10_seed42_dashboard_v2"
    / "model"
    / "nmf_pipeline.joblib"
)

SAMPLE_ARTICLES: list[dict[str, str]] = [
    {
        "title": "Ngân hàng Nhà nước điều hành linh hoạt lãi suất và tỷ giá dịp cuối năm",
        "content": (
            "Ngân hàng Nhà nước Việt Nam thông báo tiếp tục điều hành chính sách tiền tệ chủ động, "
            "giữ ổn định mặt bằng lãi suất cho vay nhằm hỗ trợ doanh nghiệp phục hồi sản xuất kinh doanh, "
            "đồng thời kiểm soát lạm phát và ổn định tỷ giá ngoại tệ trên thị trường tài chính."
        ),
        "source": "VnExpress",
        "published_at": "2026-09-18",
        "url": "",
    },
    {
        "title": "Các ngân hàng thương mại giảm lãi suất vay mua nhà và tín dụng tiêu dùng",
        "content": (
            "Nhiều ngân hàng thương mại cổ phần vừa công bố gói tín dụng ưu đãi mới với lãi suất cho vay "
            "giảm từ 0,5 đến 1% mỗi năm. Động thái này giúp khơi thông dòng vốn tín dụng vào lĩnh vực "
            "bất động sản, tiêu dùng và hỗ trợ doanh nghiệp vừa và nhỏ tiếp cận nguồn vốn giá rẻ."
        ),
        "source": "Tuổi Trẻ",
        "published_at": "2026-09-19",
        "url": "",
    },
    {
        "title": "Đội tuyển Việt Nam triệu tập lực lượng chuẩn bị cho vòng loại giải châu Á",
        "content": (
            "Huấn luyện viên trưởng đội tuyển bóng đá quốc gia Việt Nam đã công bố danh sách các cầu thủ "
            "tập trung cho loạt trận giao hữu quốc tế và vòng loại sắp tới. Nhiều tiền đạo và tiền vệ "
            "đang có phong độ cao tại giải vô địch quốc gia V-League được trao cơ hội."
        ),
        "source": "VnExpress",
        "published_at": "2026-09-20",
        "url": "",
    },
    {
        "title": "Tuyển bóng đá Việt Nam rèn chiến thuật trước trận đấu giao hữu quốc tế",
        "content": (
            "Tại buổi tập chiều nay trên sân vận động Mỹ Đình, các cầu thủ đội tuyển Việt Nam tập trung "
            "rèn luyện các bài phối hợp tấn công nhanh và dứt điểm. Ban huấn luyện đánh giá cao tinh thần "
            "thi đấu của các tuyển thủ trẻ trước thềm giải đấu bóng đá châu lục."
        ),
        "source": "VietNamNet",
        "published_at": "2026-09-21",
        "url": "",
    },
    {
        "title": "Bộ Giáo dục công bố phương án thi tốt nghiệp THPT và tuyển sinh đại học",
        "content": (
            "Bộ Giáo dục và Đào tạo vừa ban hành quy chế hướng dẫn kỳ thi tốt nghiệp trung học phổ thông "
            "và xét tuyển đại học, cao đẳng. Các trường đại học được yêu cầu công khai đề án tuyển sinh, "
            "chỉ tiêu xét tuyển học bạ và điểm thi đánh giá năng lực của thí sinh."
        ),
        "source": "Tuổi Trẻ",
        "published_at": "2026-09-22",
        "url": "",
    },
    {
        "title": "Các trường đại học công bố phương thức xét tuyển sớm và chỉ tiêu năm học mới",
        "content": (
            "Nhiều trường đại học lớn tại Hà Nội và TP.HCM đã thông báo phương án tuyển sinh đại học chính quy, "
            "kết hợp kết quả kỳ thi tốt nghiệp THPT, chứng chỉ ngoại ngữ quốc tế và kỳ thi đánh giá tư duy "
            "nhằm tạo điều kiện thuận lợi cho học sinh lớp 12."
        ),
        "source": "VnExpress",
        "published_at": "2026-09-23",
        "url": "",
    },
    {
        "title": "Bệnh viện tăng cường phòng chống dịch sốt xuất huyết khi thời tiết mưa nhiều",
        "content": (
            "Sở Y tế và các bệnh viện tuyến cuối ghi nhận số ca mắc sốt xuất huyết nhập viện điều trị "
            "có xu hướng gia tăng. Các bác sĩ khuyến cáo người dân chủ động diệt lăng quăng, phòng tránh "
            "muỗi đốt và đưa bệnh nhân đến cơ sở y tế ngay khi có triệu chứng sốt cao liên tục."
        ),
        "source": "VietNamNet",
        "published_at": "2026-09-24",
        "url": "",
    },
]


def get_csv_template_bytes() -> bytes:
    """Tạo file CSV mẫu để người dùng tải về và điền dữ liệu."""
    template = pd.DataFrame(SAMPLE_ARTICLES[:3])[
        ["title", "content", "source", "published_at", "url"]
    ]
    return template.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def _make_article_id(title: str, content: str, url: str = "", idx: int = 0) -> str:
    seed = f"{url.strip()}|{title.strip()}|{content[:120].strip()}|{idx}"
    return "new_" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]


@st.cache_resource(show_spinner=False)
def _load_shared_nlp_resources() -> dict[str, Any]:
    """Tải bộ từ dừng, vectorizer TF-IDF và từ điển tách từ ghép tiếng Việt."""
    stopwords = load_stopwords(STOPWORDS_PATH) if STOPWORDS_PATH.exists() else set()
    vectorizer = None
    compound_map: dict[str, str] = {}

    tfidf_vec_path = TFIDF_DIR / "tfidf_vectorizer.joblib"
    if tfidf_vec_path.exists():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vectorizer = joblib.load(tfidf_vec_path)
        for token in vectorizer.get_feature_names_out():
            if "_" in token:
                compound_map[token.replace("_", " ")] = token

    has_pyvi = False
    try:
        from pyvi import ViTokenizer  # noqa: F401
        has_pyvi = True
    except ImportError:
        has_pyvi = False

    return {
        "stopwords": stopwords,
        "tfidf_vectorizer": vectorizer,
        "compound_map": compound_map,
        "has_pyvi": has_pyvi,
    }


@st.cache_resource(show_spinner=False)
def _load_corpus_tfidf_index() -> dict[str, Any] | None:
    """Nạp ma trận TF-IDF của toàn bộ 9,274 bài báo trong kho để tìm bài tương tự."""
    matrices = []
    article_ids: list[str] = []
    for split in ("train", "val", "test"):
        npz_path = TFIDF_DIR / f"tfidf_{split}.npz"
        ids_path = TFIDF_DIR / f"{split}_article_ids.json"
        if npz_path.exists() and ids_path.exists():
            matrices.append(sparse.load_npz(npz_path))
            ids_data = json.loads(ids_path.read_text(encoding="utf-8"))
            if isinstance(ids_data, dict):
                ids_list = ids_data.get("article_ids", [])
            else:
                ids_list = ids_data
            article_ids.extend(str(aid) for aid in ids_list)
    if not matrices or not article_ids:
        return None
    full_matrix = sparse.vstack(matrices, format="csr")
    if full_matrix.shape[0] != len(article_ids):
        min_len = min(full_matrix.shape[0], len(article_ids))
        full_matrix = full_matrix[:min_len]
        article_ids = article_ids[:min_len]
    return {"matrix": full_matrix, "article_ids": np.array(article_ids)}


@st.cache_resource(show_spinner=False)
def _load_model_pipeline(method: str) -> Any:
    """Tải pipeline huấn luyện sẵn trên đĩa theo phương pháp."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "nmf" and NMF_PIPELINE_PATH.exists():
            return joblib.load(NMF_PIPELINE_PATH)
        if method == "lda" and LDA_PIPELINE_PATH.exists():
            return joblib.load(LDA_PIPELINE_PATH)
        if method == "kmeans" and KMEANS_MODEL_PATH.exists():
            return joblib.load(KMEANS_MODEL_PATH)
    return None


def _segment_with_compound_dict(text: str, compound_map: dict[str, str]) -> str:
    """Tách từ ghép tiếng Việt bằng thuật toán Longest-Match trên từ điển 8,786 từ ghép."""
    if not text or not compound_map:
        return text
    tokens = text.split()
    n = len(tokens)
    out: list[str] = []
    i = 0
    while i < n:
        matched = False
        for k in (4, 3, 2):
            if i + k <= n:
                phrase = " ".join(tokens[i : i + k])
                if phrase in compound_map:
                    out.append(compound_map[phrase])
                    i += k
                    matched = True
                    break
        if not matched:
            out.append(tokens[i])
            i += 1
    return " ".join(out)


def preprocess_vietnamese_texts(texts: list[str]) -> list[str]:
    """Chuẩn hóa P0 -> P1 -> Tách từ P2 -> Lọc từ dừng P3."""
    res = _load_shared_nlp_resources()
    stopwords = res["stopwords"]
    compound_map = res["compound_map"]
    has_pyvi = res["has_pyvi"]

    if has_pyvi:
        from pyvi import ViTokenizer

    processed: list[str] = []
    for raw in texts:
        p0 = clean_text_p0(raw or "")
        p1 = normalize_text_p1(p0)
        if has_pyvi and p1:
            p2 = ViTokenizer.tokenize(p1)
        else:
            p2 = _segment_with_compound_dict(p1, compound_map)
        p3 = remove_stopwords_p3(p2, stopwords) if stopwords else p2
        processed.append(p3)
    return processed


def extract_article_from_html(html_text: str, url: str = "") -> dict[str, str]:
    """Bóc tách tiêu đề, nội dung, nguồn báo và ngày xuất bản từ HTML."""
    soup = BeautifulSoup(html_text or "", "html.parser")
    for tag in soup.find_all(["script", "style", "noscript", "iframe", "header", "footer", "nav"]):
        # Giữ lại ld+json nếu cần đọc ngày đăng trước khi xóa script
        if tag.name == "script" and tag.get("type") == "application/ld+json":
            continue
        tag.decompose()

    domain = urlparse(url).netloc.lower().replace("www.", "") if url else ""
    if "vnexpress.net" in domain:
        source = "VnExpress"
    elif "tuoitre.vn" in domain:
        source = "Tuổi Trẻ"
    elif "vietnamnet.vn" in domain:
        source = "VietNamNet"
    elif domain:
        source = domain
    else:
        source = "Web"

    # 1. Tiêu đề
    title_el = (
        soup.find("h1", class_="title-detail")
        or soup.find("h1", class_="article-title")
        or soup.find("h1", class_="content-detail-title")
        or soup.find("h1", class_="article-detail-title")
        or soup.find("h1")
    )
    title = clean_text_p0(title_el.get_text(" ")) if title_el else ""
    if not title:
        og_title = soup.find("meta", {"property": "og:title"}) or soup.find("title")
        if og_title:
            title = clean_text_p0(og_title.get("content", "") if og_title.name == "meta" else og_title.get_text(" "))

    # 2. Ngày xuất bản
    published_at = ""
    meta_date = (
        soup.find("meta", {"property": "article:published_time"})
        or soup.find("meta", {"name": "pubdate"})
        or soup.find("meta", {"itemprop": "datePublished"})
    )
    if meta_date and meta_date.get("content"):
        published_at = str(meta_date["content"]).strip()[:19]

    # 3. Sapo / mô tả ngắn
    sapo_el = (
        soup.find("p", class_="description")
        or soup.find("h2", class_="sapo")
        or soup.find("h2", class_="article-sapo")
        or soup.find("div", class_="content-detail-sapo")
        or soup.find("meta", {"property": "og:description"})
        or soup.find("meta", {"name": "description"})
    )
    sapo = ""
    if sapo_el:
        sapo = clean_text_p0(
            sapo_el.get("content", "") if sapo_el.name == "meta" else sapo_el.get_text(" ")
        )

    # 4. Nội dung chính
    content_box = (
        soup.find("article", class_="fck_detail")
        or soup.find("div", class_="fck_detail")
        or soup.find("div", class_="detail-cmain")
        or soup.find("div", id="main-detail-body")
        or soup.find("div", class_="maincontent")
        or soup.find("div", class_="main-content-body")
        or soup.find("article")
        or soup.find("main")
        or soup.body
        or soup
    )
    paragraphs: list[str] = []
    seen: set[str] = set()
    if sapo:
        paragraphs.append(sapo)
        seen.add(sapo)

    for p in content_box.find_all("p"):
        if p.find_parent("figure") or p.find_parent("figcaption"):
            continue
        cls = " ".join(p.get("class") or []).lower()
        if any(skip in cls for skip in ("image", "caption", "author", "comment")):
            continue
        txt = clean_text_p0(p.get_text(" "))
        if len(txt.split()) >= 5 and txt not in seen:
            seen.add(txt)
            paragraphs.append(txt)

    content = " ".join(paragraphs).strip()
    if not content and sapo:
        content = sapo

    return {
        "title": title or (url if url else "Bài báo không có tiêu đề"),
        "content": content,
        "source": source,
        "published_at": published_at,
        "url": url,
    }


def fetch_articles_from_urls(
    urls_input: str | list[str],
    timeout: int = 10,
) -> tuple[list[dict[str, str]], list[str]]:
    """Tải và bóc tách danh sách URL bài báo. Trả về (danh_sách_bài, danh_sách_lỗi)."""
    if isinstance(urls_input, str):
        raw_urls = [
            line.strip()
            for line in re.split(r"[\r\n]+", urls_input)
            if line.strip()
        ]
    else:
        raw_urls = [str(u).strip() for u in urls_input if str(u).strip()]

    articles: list[dict[str, str]] = []
    errors: list[str] = []
    session = requests.Session()
    session.headers.update(HEADERS)

    for idx, url in enumerate(raw_urls):
        if not url.startswith(("http://", "https://")):
            errors.append(f"URL không hợp lệ (cần bắt đầu bằng http/https): {url}")
            continue
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code != 200:
                errors.append(f"Không tải được {url} (HTTP {resp.status_code})")
                continue
            parsed = extract_article_from_html(resp.text, url=url)
            if not parsed["content"] and len(parsed["title"].split()) < 4:
                errors.append(f"Không trích xuất được nội dung bài báo từ: {url}")
                continue
            parsed["article_id"] = _make_article_id(parsed["title"], parsed["content"], url, idx)
            articles.append(parsed)
        except requests.RequestException as exc:
            errors.append(f"Lỗi kết nối tới {url}: {type(exc).__name__}")

    return articles, errors


def parse_raw_text_entries(
    title: str = "",
    content: str = "",
    batch_text: str = "",
    source: str = "Nhập trực tiếp",
) -> list[dict[str, str]]:
    """Chuẩn hóa bài viết từ form Tiêu đề + Nội dung hoặc danh sách văn bản ngăn cách bằng `---`."""
    entries: list[dict[str, str]] = []
    clean_title = clean_text_p0(title)
    clean_content = clean_text_p0(content)

    if clean_title or clean_content:
        if not clean_title and clean_content:
            words = clean_content.split()
            clean_title = " ".join(words[:14]) + ("…" if len(words) > 14 else "")
        entries.append(
            {
                "article_id": _make_article_id(clean_title, clean_content, idx=0),
                "title": clean_title,
                "content": clean_content or clean_title,
                "source": source,
                "published_at": "",
                "url": "",
            }
        )

    if batch_text and batch_text.strip():
        blocks = [b.strip() for b in re.split(r"(?:\r?\n)\s*---+\s*(?:\r?\n)|(?:\r?\n){2,}", batch_text) if b.strip()]
        for idx, block in enumerate(blocks, start=len(entries)):
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if not lines:
                continue
            if len(lines) >= 2:
                b_title = clean_text_p0(lines[0])
                b_content = clean_text_p0(" ".join(lines[1:]))
            else:
                b_content = clean_text_p0(lines[0])
                words = b_content.split()
                b_title = " ".join(words[:14]) + ("…" if len(words) > 14 else "")
            entries.append(
                {
                    "article_id": _make_article_id(b_title, b_content, idx=idx),
                    "title": b_title,
                    "content": b_content,
                    "source": source,
                    "published_at": "",
                    "url": "",
                }
            )
    return entries


def parse_uploaded_file(filename: str, raw_bytes: bytes) -> list[dict[str, str]]:
    """Đọc các bài báo từ file `.csv`, `.json`, `.jsonl` hoặc `.txt`."""
    ext = Path(filename).suffix.lower()
    if ext == ".txt":
        text = raw_bytes.decode("utf-8-sig", errors="replace")
        return parse_raw_text_entries(batch_text=text, source=filename)

    if ext in {".json", ".jsonl"}:
        text = raw_bytes.decode("utf-8-sig", errors="replace").strip()
        if ext == ".jsonl" or ("\n" in text and not text.startswith("[")):
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            loaded = json.loads(text)
            records = loaded if isinstance(loaded, list) else loaded.get("articles", [loaded])
        frame = pd.DataFrame(records)
    else:
        # Mặc định đọc CSV
        try:
            frame = pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8-sig")
        except UnicodeDecodeError:
            frame = pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8", errors="replace")

    if frame.empty:
        return []

    col_map = {str(c).strip().lower(): c for c in frame.columns}
    title_col = next(
        (col_map[k] for k in ("title", "tiêu đề", "tieu_de", "headline") if k in col_map),
        None,
    )
    content_col = next(
        (
            col_map[k]
            for k in ("content", "nội dung", "noi_dung", "text", "body", "summary", "sapo")
            if k in col_map
        ),
        None,
    )
    source_col = next(
        (col_map[k] for k in ("source", "nguồn", "nguon", "domain") if k in col_map),
        None,
    )
    date_col = next(
        (col_map[k] for k in ("published_at", "publish_date", "date", "ngày đăng", "ngay_dang") if k in col_map),
        None,
    )
    url_col = next((col_map[k] for k in ("url", "link") if k in col_map), None)

    if title_col is None and content_col is None:
        # Dùng cột chuỗi đầu tiên làm nội dung
        str_cols = [c for c in frame.columns if frame[c].dtype == object]
        content_col = str_cols[0] if str_cols else frame.columns[0]

    articles: list[dict[str, str]] = []
    for idx, row in enumerate(frame.to_dict("records")):
        raw_title = str(row.get(title_col, "") or "") if title_col else ""
        raw_content = str(row.get(content_col, "") or "") if content_col else ""
        if raw_title == "nan":
            raw_title = ""
        if raw_content == "nan":
            raw_content = ""
        clean_title = clean_text_p0(raw_title)
        clean_content = clean_text_p0(raw_content)
        if not clean_title and not clean_content:
            continue
        if not clean_title:
            words = clean_content.split()
            clean_title = " ".join(words[:14]) + ("…" if len(words) > 14 else "")
        if not clean_content:
            clean_content = clean_title
        url_val = str(row.get(url_col, "") or "").strip() if url_col else ""
        if url_val == "nan":
            url_val = ""
        src_val = str(row.get(source_col, "") or "").strip() if source_col else filename
        if not src_val or src_val == "nan":
            src_val = filename
        date_val = str(row.get(date_col, "") or "").strip() if date_col else ""
        if date_val == "nan":
            date_val = ""
        articles.append(
            {
                "article_id": str(row.get("article_id") or _make_article_id(clean_title, clean_content, url_val, idx)),
                "title": clean_title,
                "content": clean_content,
                "source": src_val,
                "published_at": date_val,
                "url": url_val,
            }
        )
    return articles


def _matched_keywords_for_topic(
    processed_text: str,
    topic_id: int,
    bundle: ArtifactBundle,
    top_n: int = 5,
) -> str:
    """Tìm các từ khóa của topic xuất hiện trực tiếp trong văn bản đã xử lý."""
    if topic_id == -1:
        return ""
    sub = bundle.topic_keywords.loc[bundle.topic_keywords["topic_id"] == topic_id].head(20)
    if sub.empty:
        return ""
    tokens = set(processed_text.split())
    plain_text = f" {processed_text.replace('_', ' ')} "
    matched: list[str] = []
    for kw in sub["keyword"].astype(str):
        kw_clean = kw.strip()
        if kw_clean in tokens or kw_clean.replace(" ", "_") in tokens or f" {kw_clean.replace('_', ' ')} " in plain_text:
            matched.append(kw_clean.replace("_", " "))
        if len(matched) >= top_n:
            break
    if not matched:
        matched = [str(k).replace("_", " ") for k in sub["keyword"].head(3)]
    return ", ".join(matched)


def assign_topics_to_articles(
    articles: list[dict[str, str]] | pd.DataFrame,
    bundle: ArtifactBundle,
    run: ArtifactRun,
    labels: dict[int, str],
    outlier_threshold: float = 0.08,
) -> pd.DataFrame:
    """Gán mỗi bài báo đầu vào vào hệ thống Chủ đề của `run` đang chọn."""
    frame = pd.DataFrame(articles).copy() if not isinstance(articles, pd.DataFrame) else articles.copy()
    if frame.empty:
        return frame

    for col in ("article_id", "title", "content", "source", "published_at", "url"):
        if col not in frame.columns:
            frame[col] = ""

    combined_raw = [
        f"{t}. {c}".strip()
        for t, c in zip(frame["title"].fillna(""), frame["content"].fillna(""))
    ]
    processed_texts = preprocess_vietnamese_texts(combined_raw)
    frame["processed_text"] = processed_texts

    method = run.method.lower()
    active_topic_ids = sorted(
        int(tid) for tid in bundle.topic_summary["topic_id"].unique() if int(tid) != -1
    )
    if not active_topic_ids:
        active_topic_ids = [0]

    # Ma trận điểm xác suất (n_docs, n_topics)
    score_matrix = np.zeros((len(frame), len(active_topic_ids)), dtype=np.float64)
    topic_idx_map = {tid: i for i, tid in enumerate(active_topic_ids)}

    pipeline = _load_model_pipeline(method)
    res = _load_shared_nlp_resources()
    tfidf_vectorizer = res["tfidf_vectorizer"]

    if method == "nmf" and isinstance(pipeline, dict) and "vectorizer" in pipeline:
        X = pipeline["vectorizer"].transform(processed_texts)
        W = pipeline["topic_model"].transform(X)
        for tid in active_topic_ids:
            if tid < W.shape[1]:
                score_matrix[:, topic_idx_map[tid]] = W[:, tid]
    elif method == "lda" and isinstance(pipeline, dict) and "vectorizer" in pipeline:
        X = pipeline["vectorizer"].transform(processed_texts)
        W = pipeline["lda_model"].transform(X)
        for tid in active_topic_ids:
            if tid < W.shape[1]:
                score_matrix[:, topic_idx_map[tid]] = W[:, tid]
    elif method == "kmeans" and pipeline is not None and tfidf_vectorizer is not None:
        X = tfidf_vectorizer.transform(processed_texts)
        dists = pipeline.transform(X)
        # Chuyển khoảng cách Euclid sang điểm tương đồng mềm
        sim = np.exp(-2.5 * dists)
        row_sums = sim.sum(axis=1, keepdims=True)
        probs = np.divide(sim, row_sums, out=np.zeros_like(sim), where=row_sums > 0)
        for tid in active_topic_ids:
            if tid < probs.shape[1]:
                score_matrix[:, topic_idx_map[tid]] = probs[:, tid]
    else:
        # BERTopic / HDBSCAN (hoặc fallback tổng quát):
        # Kết hợp k-NN Cosine Similarity trên kho 9,274 bài + khớp hồ sơ từ khóa topic_keywords.csv
        corpus_idx = _load_corpus_tfidf_index()
        if tfidf_vectorizer is not None and corpus_idx is not None:
            X_query = tfidf_vectorizer.transform(processed_texts)
            sims = (X_query @ corpus_idx["matrix"].T).toarray()
            doc_topic_map = dict(
                zip(
                    bundle.document_topics["article_id"].astype(str),
                    bundle.document_topics["topic_id"].astype(int),
                )
            )
            corpus_topics = np.array(
                [doc_topic_map.get(aid, -1) for aid in corpus_idx["article_ids"]],
                dtype=int,
            )
            for row_i in range(len(frame)):
                row_sim = sims[row_i]
                top_k_idx = np.argpartition(row_sim, -25)[-25:]
                for idx_n in top_k_idx:
                    sim_val = float(row_sim[idx_n])
                    t_id = int(corpus_topics[idx_n])
                    if sim_val > 0 and t_id in topic_idx_map:
                        score_matrix[row_i, topic_idx_map[t_id]] += sim_val

        # Cộng điểm khớp trực tiếp với từ khóa của từng topic
        for tid in active_topic_ids:
            sub_kw = bundle.topic_keywords.loc[bundle.topic_keywords["topic_id"] == tid].head(15)
            kw_weights = [
                (str(r["keyword"]).strip().lower(), float(r["weight"]))
                for r in sub_kw.to_dict("records")
            ]
            for row_i, p_text in enumerate(processed_texts):
                plain = f" {p_text.lower().replace('_', ' ')} "
                tokens = set(p_text.lower().split())
                kw_score = 0.0
                for kw, weight in kw_weights:
                    if kw in tokens or kw.replace(" ", "_") in tokens or f" {kw.replace('_', ' ')} " in plain:
                        kw_score += max(weight, 0.05)
                score_matrix[row_i, topic_idx_map[tid]] += kw_score * 1.5

    supports_outlier = method in {"bertopic", "hdbscan"}
    assigned_ids: list[int] = []
    assigned_labels: list[str] = []
    confidences: list[float] = []
    top3_texts: list[str] = []
    matched_kw_list: list[str] = []

    for row_i in range(len(frame)):
        scores = score_matrix[row_i]
        total = float(scores.sum())
        max_raw = float(scores.max()) if len(scores) else 0.0
        probs = scores / total if total > 0 else np.zeros_like(scores)
        order = np.argsort(probs)[::-1]

        best_idx = int(order[0])
        best_tid = active_topic_ids[best_idx]
        best_prob = float(probs[best_idx]) if total > 0 else 0.0

        if (total <= 1e-9) or (supports_outlier and max_raw < outlier_threshold):
            final_tid = -1 if supports_outlier else best_tid
            final_prob = 0.0 if final_tid == -1 else best_prob
        else:
            final_tid = best_tid
            final_prob = best_prob

        assigned_ids.append(final_tid)
        assigned_labels.append("Outlier (Ngoại lai)" if final_tid == -1 else labels.get(final_tid, f"T{final_tid}"))
        confidences.append(round(final_prob, 4))

        top3_items = []
        for idx_rank in order[:3]:
            tid_r = active_topic_ids[int(idx_rank)]
            p_r = float(probs[int(idx_rank)])
            if p_r > 0:
                lbl_r = labels.get(tid_r, f"T{tid_r}")
                top3_items.append(f"{lbl_r} ({p_r:.1%})")
        top3_texts.append(" · ".join(top3_items) if top3_items else "Chưa rõ")
        matched_kw_list.append(_matched_keywords_for_topic(processed_texts[row_i], final_tid, bundle))

    frame["topic_id"] = assigned_ids
    frame["topic_label"] = assigned_labels
    frame["confidence"] = confidences
    frame["top_3_topics"] = top3_texts
    frame["matched_keywords"] = matched_kw_list
    return frame


def cluster_articles_into_events(
    articles_df: list[dict[str, str]] | pd.DataFrame,
    similarity_threshold: float = 0.14,
    n_clusters: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tự động gom cụm các bài báo đầu vào thành các Nhóm Sự kiện cụ thể (Dynamic Event Clustering).

    Trả về:
        - `articles_with_events`: DataFrame bài báo kèm `event_id`, `event_label`, `event_similarity`, `is_centroid`, `x`, `y`.
        - `event_summary`: Bảng tổng hợp các nhóm sự kiện phát hiện được.
    """
    frame = (
        pd.DataFrame(articles_df).copy()
        if not isinstance(articles_df, pd.DataFrame)
        else articles_df.copy()
    )
    if frame.empty:
        return frame, pd.DataFrame()

    if "processed_text" not in frame.columns:
        combined_raw = [
            f"{t}. {c}".strip()
            for t, c in zip(frame["title"].fillna(""), frame["content"].fillna(""))
        ]
        frame["processed_text"] = preprocess_vietnamese_texts(combined_raw)

    n_docs = len(frame)
    local_vec = TfidfVectorizer(
        ngram_range=(1, 1),
        token_pattern=r"(?u)\b\w+\b",
        max_features=5000,
        sublinear_tf=True,
    )
    try:
        X_local = local_vec.fit_transform(frame["processed_text"].fillna(""))
        feature_names = local_vec.get_feature_names_out()
    except ValueError:
        X_local = sparse.csr_matrix((n_docs, 1), dtype=np.float64)
        feature_names = np.array(["chủ_đề"])

    sim_matrix = np.clip(cosine_similarity(X_local), 0.0, 1.0)
    dist_matrix = np.clip(1.0 - sim_matrix, 0.0, 1.0)
    np.fill_diagonal(dist_matrix, 0.0)

    if n_docs == 1:
        cluster_labels = np.array([0], dtype=int)
    elif n_clusters is not None and 1 <= n_clusters <= n_docs:
        if n_clusters == 1:
            cluster_labels = np.zeros(n_docs, dtype=int)
        elif n_clusters == n_docs:
            cluster_labels = np.arange(n_docs, dtype=int)
        else:
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric="precomputed",
                linkage="average",
            )
            cluster_labels = model.fit_predict(dist_matrix)
    else:
        distance_threshold = float(np.clip(1.0 - similarity_threshold, 0.05, 0.95))
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="precomputed",
            linkage="average",
        )
        cluster_labels = model.fit_predict(dist_matrix)

    # Sắp xếp lại ID sự kiện theo quy mô giảm dần (0, 1, 2, ...)
    unique_ids, counts = np.unique(cluster_labels, return_counts=True)
    sorted_old_ids = [int(uid) for uid, _ in sorted(zip(unique_ids, counts), key=lambda x: (-x[1], x[0]))]
    remap = {old_id: new_id for new_id, old_id in enumerate(sorted_old_ids)}
    event_ids = np.array([remap[int(c)] for c in cluster_labels], dtype=int)

    # Tính tọa độ 2D cho bản đồ trực quan
    if n_docs >= 3 and X_local.shape[1] >= 2:
        svd = TruncatedSVD(n_components=2, random_state=42)
        coords = svd.fit_transform(X_local)
        frame["x"] = coords[:, 0]
        frame["y"] = coords[:, 1]
    elif n_docs == 2:
        frame["x"] = [-0.5, 0.5]
        frame["y"] = [0.0, 0.0]
    else:
        frame["x"] = [0.0]
        frame["y"] = [0.0]

    event_labels_map: dict[int, str] = {}
    event_keywords_map: dict[int, str] = {}
    centroid_flags = np.zeros(n_docs, dtype=bool)
    member_sims = np.ones(n_docs, dtype=np.float64)
    summary_rows: list[dict[str, Any]] = []

    for eid in range(len(sorted_old_ids)):
        idx_list = np.where(event_ids == eid)[0]
        sub_X = X_local[idx_list]
        mean_tfidf = np.asarray(sub_X.mean(axis=0)).ravel()
        top_feat_idx = np.argsort(mean_tfidf)[::-1][:6]
        kws: list[str] = []
        for f_i in top_feat_idx:
            if mean_tfidf[f_i] > 0:
                word = str(feature_names[f_i]).replace("_", " ").strip()
                if word and not any(word in existing or existing in word for existing in kws):
                    kws.append(word)
            if len(kws) >= 4:
                break
        if not kws:
            kws = ["tin tức"]

        short_name = " · ".join(kws[:3])
        event_label = f"E{eid} · {short_name}"
        event_labels_map[eid] = event_label
        event_keywords_map[eid] = ", ".join(kws)

        # Bài viết tâm điểm (có độ tương đồng trung bình cao nhất với các bài trong nhóm)
        sub_sim = sim_matrix[np.ix_(idx_list, idx_list)]
        avg_sims = sub_sim.mean(axis=1)
        best_local = int(np.argmax(avg_sims))
        centroid_doc_idx = int(idx_list[best_local])
        centroid_flags[centroid_doc_idx] = True
        for local_i, doc_idx in enumerate(idx_list):
            member_sims[int(doc_idx)] = round(float(avg_sims[local_i]), 4)

        summary_rows.append(
            {
                "event_id": eid,
                "event_label": event_label,
                "document_count": len(idx_list),
                "percentage": len(idx_list) / n_docs if n_docs else 0.0,
                "top_keywords": ", ".join(kws),
                "representative_title": str(frame.iloc[centroid_doc_idx]["title"]),
                "mean_similarity": round(float(avg_sims.mean()), 4),
            }
        )

    frame["event_id"] = event_ids
    frame["event_label"] = [event_labels_map[int(e)] for e in event_ids]
    frame["event_keywords"] = [event_keywords_map[int(e)] for e in event_ids]
    frame["event_similarity"] = member_sims
    frame["is_centroid"] = centroid_flags

    return frame, pd.DataFrame(summary_rows)


def find_related_corpus_articles(
    processed_texts: list[str],
    bundle: ArtifactBundle,
    labels: dict[int, str],
    top_k: int = 5,
) -> pd.DataFrame:
    """Tìm các bài báo tương đồng nhất trong kho 9,274 bài của dự án dựa trên trung tâm TF-IDF."""
    res = _load_shared_nlp_resources()
    tfidf_vectorizer = res["tfidf_vectorizer"]
    corpus_idx = _load_corpus_tfidf_index()
    if tfidf_vectorizer is None or corpus_idx is None or not processed_texts:
        return pd.DataFrame()

    X_query = tfidf_vectorizer.transform(processed_texts)
    mean_query = sparse.csr_matrix(X_query.mean(axis=0))
    sims = (mean_query @ corpus_idx["matrix"].T).toarray().ravel()
    if sims.max() <= 0:
        return pd.DataFrame()

    top_indices = np.argsort(sims)[::-1][:top_k]
    matched_ids = [str(corpus_idx["article_ids"][i]) for i in top_indices if sims[i] > 0]
    sim_by_id = {str(corpus_idx["article_ids"][i]): float(sims[i]) for i in top_indices if sims[i] > 0}
    if not matched_ids:
        return pd.DataFrame()

    docs = enrich_documents(bundle.document_topics)
    sub = docs.loc[docs["article_id"].astype(str).isin(matched_ids)].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["similarity"] = sub["article_id"].astype(str).map(sim_by_id).fillna(0.0)
    sub["topic_label"] = [
        "Outlier" if int(t) == -1 else labels.get(int(t), f"T{int(t)}")
        for t in sub["topic_id"]
    ]
    return sub.sort_values("similarity", ascending=False).reset_index(drop=True)
