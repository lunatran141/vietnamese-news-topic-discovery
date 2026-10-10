"""
Module tiền xử lý mức P3 — Stopword Removal (Task D03).

Mục đích:
  - Loại bỏ các từ dừng tiếng Việt dựa trên danh sách chuẩn có kiểm soát.
  - Bảo vệ tuyệt đối các từ phủ định ('không', 'chưa', 'chẳng'...) và từ khóa chuyên mục.
  - Đảm bảo cơ chế chống rỗng chuỗi (Non-empty Guard).
  - Phục vụ thực nghiệm bóc tách cho các mô hình túi từ truyền thống (LDA, NMF).

Hàm chính:
  remove_stopwords_p3(text: str, stopwords: set | None = None, stopwords_path: Path | str | None = None) -> str
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Set

logger = logging.getLogger("remove_stopwords")

# Danh sách các từ phủ định và từ khóa chủ đề tối quan trọng không xóa
PROTECTED_WORDS = {
    "không", "chưa", "chẳng", "chả",
    "không_thể", "chưa_thể", "chẳng_thể", "không_phải", "chưa_phải",
    "kinh_doanh", "công_nghệ", "thể_thao", "giáo_dục", "sức_khỏe", "pháp_luật", "giải_trí",
}

# Cache tập từ dừng trong bộ nhớ để tránh đọc lại tệp nhiều lần
_STOPWORDS_CACHE: Set[str] | None = None


def load_stopwords(stopwords_path: Path | str | None = None) -> Set[str]:
    """
    Tải danh sách từ dừng tiếng Việt từ file txt.
    Tự động loại bỏ các từ thuộc danh sách bảo vệ PROTECTED_WORDS ra khỏi tập từ dừng.
    """
    global _STOPWORDS_CACHE
    if _STOPWORDS_CACHE is not None and stopwords_path is None:
        return _STOPWORDS_CACHE

    if stopwords_path is None:
        root = Path(__file__).resolve().parents[2]
        stopwords_path = root / "configs" / "vietnamese_stopwords.txt"

    p = Path(stopwords_path)
    sw_set: Set[str] = set()

    if p.exists():
        with open(p, encoding="utf-8") as f:
            for line in f:
                w = line.strip().lower()
                # Bỏ qua dòng trống hoặc dòng chú thích bắt đầu bằng '#'
                if not w or w.startswith("#"):
                    continue
                # Bảo vệ từ phủ định và từ khóa chủ đề
                if w not in PROTECTED_WORDS:
                    sw_set.add(w)
    else:
        logger.warning("Không tìm thấy file từ dừng: %s. Sử dụng danh sách rỗng.", p)

    if stopwords_path is None:
        _STOPWORDS_CACHE = sw_set

    return sw_set


def remove_stopwords_p3(
    text: str,
    stopwords: Set[str] | None = None,
    stopwords_path: Path | str | None = None,
) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    if stopwords is None:
        stopwords = load_stopwords(stopwords_path)
    tokens = text.split()
    if not tokens:
        return ""

    # Lọc bỏ từ dừng nhưng bảo vệ từ phủ định và từ khóa chủ đề
    filtered = [
        tok for tok in tokens
        if tok.lower() not in stopwords or tok.lower() in PROTECTED_WORDS
    ]

    # Cơ chế chống rỗng chuỗi (Non-empty Guard)
    if not filtered:
        logger.debug("Văn bản bị rỗng sau khi lọc stopword, hoàn trả văn bản gốc P2.")
        return text.strip()

    return re.sub(r"\s+", " ", " ".join(filtered)).strip()

