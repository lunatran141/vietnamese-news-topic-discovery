"""
Module tiền xử lý mức P1 — Controlled Normalization (Task D03).

Mục đích:
  - Chuẩn hóa có kiểm soát: chuyển chữ thường (lowercase), đồng nhất dấu nháy/gạch.
  - Thay thế URL bằng token URL, Email bằng token EMAIL.
  - Loại bỏ các đoạn boilerplate báo chí (quảng cáo, "xem thêm", "chia sẻ", "theo dõi").
  - Xóa tiêu đề bị lặp lại ở đầu bài viết nội dung.
  - Phục vụ làm đầu vào sạch nhiễu cho Embeddings và TF-IDF có kiểm soát.

Hàm chính:
  normalize_text_p1(text: str, title: str = "") -> str
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

import yaml

logger = logging.getLogger("normalize_text")

# Các mẫu regex loại bỏ boilerplate báo chí mặc định
DEFAULT_BOILERPLATE_PATTERNS = [
    r"(?i)\b(xem\s+thêm|đọc\s+tiếp|xem\s+tiếp)\s*:?.*$",
    r"(?i)\b(theo\s+dõi|chia\s+sẻ)\s+(chúng\s+tôi\s+)?(trên|qua)\s+(facebook|zalo|tiktok|youtube|google\s+news).*$",
    r"(?i)\b(bấm\s+vào\s+đây|click\s+vào\s+đây|liên\s+hệ\s+quảng\s+cáo|mọi\s+ý\s+kiến\s+đóng\s+góp).*$",
    r"(?i)\b(bản\s+quyền\s+thuộc\s+về|nguồn\s+tin\s*:|theo\s+nguồn\s*:).*$",
    r"(?i)\b(ảnh\s*:|video\s*:|đồ\s+họa\s*:)\s*[^.\n]+$",
]


def load_boilerplate_patterns(config_path: Path | str | None = None) -> List[str]:
    """Tải các mẫu regex boilerplate từ file cấu hình preprocessing.yaml nếu có."""
    if config_path is not None:
        p = Path(config_path)
        if p.exists():
            try:
                with open(p, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                patterns = cfg.get("p1_normalized", {}).get("boilerplate_patterns", [])
                if patterns:
                    return patterns
            except Exception as e:
                logger.warning("Không thể đọc config boilerplate: %s. Dùng mẫu mặc định.", e)
    return DEFAULT_BOILERPLATE_PATTERNS


def normalize_text_p1(text: str, title: str = "", config_path: Path | str | None = None) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    original_len = len(text)

    # Chuyển về chữ thường
    normalized = text.lower()

    # Đồng nhất dấu nháy và dấu gạch
    normalized = re.sub(r"[“”„‟«»]", '"', normalized)
    normalized = re.sub(r"[‘’‚‛]", "'", normalized)
    normalized = re.sub(r"[–—−]", "-", normalized)

    # Thay thế URL và Email bằng token đặc biệt
    normalized = re.sub(r"https?://\S+|www\.\S+", " URL ", normalized)
    normalized = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", " EMAIL ", normalized)

    # Xóa tiêu đề bị lặp lại ở đầu bài viết nội dung (nếu có)
    if title and isinstance(title, str) and title.strip():
        norm_title = title.lower().strip()
        norm_title = re.sub(r"[“”„‟«»]", '"', norm_title)
        norm_title = re.sub(r"[‘’‚‛]", "'", norm_title)
        if normalized.startswith(norm_title):
            normalized = normalized[len(norm_title):].strip()
            normalized = re.sub(r"^[\s\-:–—]+", "", normalized).strip()

    # Loại bỏ boilerplate theo các mẫu regex
    patterns = load_boilerplate_patterns(config_path)
    for pat in patterns:
        normalized = re.sub(pat, " ", normalized)

    # Chuẩn hóa khoảng trắng liên tiếp và cắt tỉa
    normalized = re.sub(r"\s+", " ", normalized).strip()

    # Cảnh báo nếu văn bản bị giảm hơn 20% độ dài
    new_len = len(normalized)
    if original_len > 100 and (original_len - new_len) / original_len > 0.20:
        logger.debug("Bài viết giảm > 20%% độ dài sau khi lọc boilerplate (%d -> %d)", original_len, new_len)

    if not normalized and text.strip():
        return text.lower().strip()

    return normalized

