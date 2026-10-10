"""
Module tiền xử lý mức P0 — Minimal Cleaning (Task D03).

Mục đích:
  - Làm sạch tối thiểu, chuẩn hóa mã Unicode NFC, gỡ bỏ HTML và ký tự điều khiển.
  - Bảo tồn nguyên vẹn chữ hoa/chữ thường, dấu câu tự nhiên và ngữ pháp.
  - Phục vụ làm đầu vào cho BERTopic, Transformer / LLMs và Word Embeddings.

Hàm chính:
  clean_text_p0(text: str) -> str
"""
from __future__ import annotations

import html
import re
import unicodedata


def clean_text_p0(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    # Giải mã HTML entities
    cleaned = html.unescape(text)

    # Loại bỏ thẻ HTML tags
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)

    # Chuẩn hóa Unicode NFC
    cleaned = unicodedata.normalize("NFC", cleaned)

    # Loại bỏ ký tự điều khiển và null bytes (giữ lại khoảng trắng)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", cleaned)

    # Chuẩn hóa khoảng trắng liên tiếp và cắt tỉa 2 đầu
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned and text.strip():
        return text.strip()

    return cleaned

