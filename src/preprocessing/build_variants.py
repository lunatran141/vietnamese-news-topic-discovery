"""
Script điều phối sinh 4 biến thể tiền xử lý văn bản (Task D03 — Build Preprocessing Variants).


Cách chạy:
    python -m src.preprocessing.build_variants
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.preprocessing.clean_text import clean_text_p0
from src.preprocessing.normalize_text import normalize_text_p1
from src.preprocessing.remove_stopwords import load_stopwords, remove_stopwords_p3
from src.preprocessing.segment_words import segment_words_p2

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = ROOT / "data" / "interim" / "news_articles_deduplicated.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
STATS_CSV = ROOT / "results" / "preprocessing_data" / "preprocessing_before_after.csv"
CONFIG_FILE = ROOT / "configs" / "preprocessing.yaml"
STOPWORDS_FILE = ROOT / "configs" / "vietnamese_stopwords.txt"

EXPECTED_COLS = [
    "article_id",
    "title",
    "content",
    "processed_title",
    "processed_content",
    "text",
    "published_at",
    "source",
    "url",
    "category_gold",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_variants")


def compute_variant_stats(name: str, texts: List[str], ref_tokens: int | None = None) -> Dict[str, Any]:
    """Tính toán các chỉ số thống kê (mean, median, p5, p95, vocab size) cho một biến thể."""
    n_docs = len(texts)
    char_lens = np.array([len(t) for t in texts])
    word_lens = np.array([len(t.split()) for t in texts])
    
    # Đếm từ vựng độc nhất
    vocab = set()
    total_tokens = int(np.sum(word_lens))
    for t in texts:
        vocab.update(t.split())

    empty_count = int(np.sum(char_lens == 0))
    retention_pct = 100.0 if ref_tokens is None or ref_tokens == 0 else round((total_tokens / ref_tokens) * 100, 2)

    return {
        "variant": name,
        "total_articles": n_docs,
        "empty_text_count": empty_count,
        "vocab_size": len(vocab),
        "total_tokens": total_tokens,
        "retention_rate_pct": retention_pct,
        "mean_length_chars": round(float(np.mean(char_lens)), 2),
        "median_length_chars": round(float(np.median(char_lens)), 2),
        "p5_length_chars": round(float(np.percentile(char_lens, 5)), 2),
        "p95_length_chars": round(float(np.percentile(char_lens, 95)), 2),
        "mean_word_count": round(float(np.mean(word_lens)), 2),
        "median_word_count": round(float(np.median(word_lens)), 2),
        "p5_word_count": round(float(np.percentile(word_lens, 5)), 2),
        "p95_word_count": round(float(np.percentile(word_lens, 95)), 2),
    }


def build_variants(
    input_path: Path | None = None,
    output_dir: Path | None = None,
    stats_path: Path | None = None,
) -> Dict[str, pd.DataFrame]:
    """Thực thi tuần tự toàn bộ quy trình tiền xử lý P0 -> P1 -> P2 -> P3."""
    input_path = input_path or INPUT_CSV
    output_dir = output_dir or OUTPUT_DIR
    stats_path = stats_path or STATS_CSV

    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file đầu vào: {input_path}")

    logger.info("Nạp dữ liệu đầu vào: %s", input_path.relative_to(ROOT))
    df_raw = pd.read_csv(input_path, dtype=str)
    
    # Đảm bảo bài viết được sắp xếp tăng dần theo article_id để đồng nhất
    df_raw = df_raw.sort_values("article_id").reset_index(drop=True)
    n_articles = len(df_raw)
    logger.info("Tổng số bài báo cần xử lý: %d", n_articles)

    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    # Nạp danh sách từ dừng tuyển chọn
    stopwords_set = load_stopwords(STOPWORDS_FILE)
    logger.info("Đã nạp %d từ dừng từ %s", len(stopwords_set), STOPWORDS_FILE.name)

    titles = df_raw["title"].fillna("").tolist()
    contents = df_raw["content"].fillna("").tolist()

    # Mảng lưu kết quả văn bản qua 4 biến thể
    p0_titles, p0_contents, p0_texts = [], [], []
    p1_titles, p1_contents, p1_texts = [], [], []
    p2_titles, p2_contents, p2_texts = [], [], []
    p3_titles, p3_contents, p3_texts = [], [], []

    logger.info("Bắt đầu xử lý chuỗi biến thể P0 -> P1 -> P2 -> P3...")
    for i in tqdm(range(n_articles), desc="Preprocessing"):
        raw_t = titles[i]
        raw_c = contents[i]

        # 1. P0: Minimal Cleaning
        t0 = clean_text_p0(raw_t)
        c0 = clean_text_p0(raw_c)
        text0 = f"{t0} {c0}".strip()
        p0_titles.append(t0)
        p0_contents.append(c0)
        p0_texts.append(text0)

        # 2. P1: Controlled Normalization (kế thừa từ P0)
        t1 = normalize_text_p1(t0, config_path=CONFIG_FILE)
        c1 = normalize_text_p1(c0, title=t0, config_path=CONFIG_FILE)
        text1 = f"{t1} {c1}".strip()
        p1_titles.append(t1)
        p1_contents.append(c1)
        p1_texts.append(text1)

        # 3. P2: Word Segmentation (kế thừa từ P1)
        t2 = segment_words_p2(t1)
        c2 = segment_words_p2(c1)
        text2 = f"{t2} {c2}".strip()
        p2_titles.append(t2)
        p2_contents.append(c2)
        p2_texts.append(text2)

        # 4. P3: Stopword Removal (kế thừa từ P2)
        t3 = remove_stopwords_p3(t2, stopwords=stopwords_set)
        c3 = remove_stopwords_p3(c2, stopwords=stopwords_set)
        text3 = f"{t3} {c3}".strip()
        p3_titles.append(t3)
        p3_contents.append(c3)
        p3_texts.append(text3)

    # Tạo 4 DataFrame độc lập bảo toàn cấu trúc metadata
    def make_variant_df(t_list: List[str], c_list: List[str], text_list: List[str]) -> pd.DataFrame:
        df_var = df_raw.copy()
        df_var["processed_title"] = t_list
        df_var["processed_content"] = c_list
        df_var["text"] = text_list
        return df_var[EXPECTED_COLS].copy()

    df_p0 = make_variant_df(p0_titles, p0_contents, p0_texts)
    df_p1 = make_variant_df(p1_titles, p1_contents, p1_texts)
    df_p2 = make_variant_df(p2_titles, p2_contents, p2_texts)
    df_p3 = make_variant_df(p3_titles, p3_contents, p3_texts)

    # Lưu 4 tệp dữ liệu biến thể với encoding utf-8-sig
    files_map = {
        "p0_minimal.csv": df_p0,
        "p1_normalized.csv": df_p1,
        "p2_segmented.csv": df_p2,
        "p3_no_stopwords.csv": df_p3,
    }

    for filename, df_v in files_map.items():
        out_f = output_dir / filename
        df_v.to_csv(out_f, index=False, encoding="utf-8-sig")
        logger.info("Đã lưu %s: %d dòng, %d cột", out_f.name, len(df_v), len(df_v.columns))

    # Tính toán bảng thống kê Before / After (mục 7.8 của đặc tả)
    logger.info("Tính toán bảng thống kê trước và sau tiền xử lý...")
    raw_texts = [f"{t} {c}".strip() for t, c in zip(titles, contents)]
    ref_total_tokens = int(np.sum([len(t.split()) for t in raw_texts]))

    stats_records = [
        compute_variant_stats("original", raw_texts, ref_total_tokens),
        compute_variant_stats("p0_minimal", p0_texts, ref_total_tokens),
        compute_variant_stats("p1_normalized", p1_texts, ref_total_tokens),
        compute_variant_stats("p2_segmented", p2_texts, ref_total_tokens),
        compute_variant_stats("p3_no_stopwords", p3_texts, ref_total_tokens),
    ]

    stats_df = pd.DataFrame(stats_records)
    stats_df.to_csv(stats_path, index=False, encoding="utf-8-sig")
    logger.info("Đã lưu bảng thống kê: %s", stats_path.relative_to(ROOT))

    return {
        "p0": df_p0,
        "p1": df_p1,
        "p2": df_p2,
        "p3": df_p3,
        "stats": stats_df,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sinh 4 biến thể tiền xử lý văn bản Task D03 (P0 -> P3)")
    p.add_argument("--input", type=Path, default=INPUT_CSV, help="Đường dẫn file đầu vào sau dedup")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Thư mục xuất 4 file processed")
    p.add_argument("--stats-path", type=Path, default=STATS_CSV, help="Đường dẫn xuất file thống kê")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        build_variants(
            input_path=args.input,
            output_dir=args.output_dir,
            stats_path=args.stats_path,
        )
        logger.info("Hoàn thành toàn bộ Task D03 thành công.")
    except Exception as e:
        logger.error("Quá trình tiền xử lý gặp lỗi: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

