"""
Module tiền xử lý mức P2 — Vietnamese Word Segmentation (Task D03).

Mục đích:
  - Tách từ tiếng Việt thống nhất cho cả tiêu đề và nội dung bằng thư viện pyvi.
  - Nối các từ ghép có nghĩa bằng dấu gạch dưới '_' (ví dụ: 'kinh tế' -> 'kinh_tế').
  - Phục vụ làm đầu vào cho các mô hình túi từ (Bag-of-Words): TF-IDF, K-Means, NMF, LDA.

Hàm chính:
  segment_words_p2(text: str) -> str
"""
from __future__ import annotations

import re
from pyvi import ViTokenizer

def segment_words_p2(text: str) -> str:
    """
    Tiền xử lý tách từ tiếng Việt (P2):
      1. Nhận văn bản đầu vào (thường là kết quả từ P1)
      2. Sử dụng pyvi.ViTokenizer.tokenize để nhận diện từ ghép tiếng Việt
      3. Nối các từ ghép bằng dấu gạch dưới '_'
      4. Chuẩn hóa khoảng trắng và cắt tỉa 2 đầu.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Tách từ tiếng Việt qua pyvi
    segmented = ViTokenizer.tokenize(text)

    # Gộp khoảng trắng thừa
    segmented = re.sub(r"\s+", " ", segmented).strip()

    if not segmented and text.strip():
        return text.strip()
    return segmented