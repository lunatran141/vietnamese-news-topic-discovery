"""Module trích xuất đặc trưng TF-IDF cho bài báo tiếng Việt.

Cung cấp pipeline xây dựng, tinh chỉnh và lưu trữ đặc trưng TF-IDF chuẩn hóa.
Tuân thủ nguyên tắc fit duy nhất trên tập train, transform nhất quán cho val/test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


def load_stopwords(stopwords_path: str | Path | None) -> list[str] | None:
    """Đọc danh sách từ dừng từ file văn bản UTF-8.
    
    Bỏ qua các dòng chú thích bắt đầu bằng '#' và các dòng trống.
    """
    if stopwords_path is None:
        return None

    path = Path(stopwords_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file từ dừng: {path}")

    stopwords: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            cleaned = line.strip()
            if cleaned and not cleaned.startswith("#"):
                stopwords.append(cleaned)

    return stopwords if stopwords else None


def create_tfidf_vectorizer(config: dict[str, Any]) -> TfidfVectorizer:
    """Khởi tạo TfidfVectorizer từ từ điển cấu hình."""
    vec_cfg = config.get("vectorizer", config)

    raw_ngram = vec_cfg.get("ngram_range", (1, 2))
    ngram_range = tuple(raw_ngram) if isinstance(raw_ngram, (list, tuple)) else (1, 2)

    min_df = vec_cfg.get("min_df", 3)
    max_df = vec_cfg.get("max_df", 0.85)
    max_features = vec_cfg.get("max_features", 15000)
    sublinear_tf = bool(vec_cfg.get("sublinear_tf", True))
    norm = vec_cfg.get("norm", "l2")
    token_pattern = vec_cfg.get("token_pattern", r"(?u)\b\w\w+\b")
    lowercase = bool(vec_cfg.get("lowercase", True))

    stopwords_file = vec_cfg.get("stopwords_file")
    stop_words = load_stopwords(stopwords_file) if stopwords_file else None

    return TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        norm=norm,
        token_pattern=token_pattern,
        lowercase=lowercase,
        stop_words=stop_words,
    )


def fit_tfidf_pipeline(
    texts: Iterable[str],
    config: dict[str, Any],
) -> TfidfVectorizer:
    """Huấn luyện (fit) vectorizer duy nhất trên tập văn bản (thường là train)."""
    text_list = list(texts)
    if not text_list:
        raise ValueError("Danh sách văn bản huấn luyện rỗng")

    vectorizer = create_tfidf_vectorizer(config)
    vectorizer.fit(text_list)

    feature_names = vectorizer.get_feature_names_out()
    if len(feature_names) == 0:
        raise ValueError("Vectorizer không học được từ vựng nào từ dữ liệu đầu vào")

    return vectorizer


def transform_tfidf_pipeline(
    vectorizer: TfidfVectorizer,
    texts: Iterable[str],
) -> csr_matrix:
    """Biến đổi tập văn bản sang ma trận thưa TF-IDF bằng vectorizer đã fit."""
    text_list = list(texts)
    if not text_list:
        raise ValueError("Danh sách văn bản biến đổi rỗng")

    matrix = vectorizer.transform(text_list)
    if not isinstance(matrix, csr_matrix):
        matrix = matrix.tocsr()

    return matrix


def evaluate_validation_vocabulary(
    vectorizer: TfidfVectorizer,
    val_texts: Iterable[str],
    val_article_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Đánh giá chất lượng biểu diễn và độ phủ từ vựng trên tập Validation."""
    text_list = list(texts if isinstance(texts := val_texts, list) else list(val_texts))
    num_docs = len(text_list)
    if num_docs == 0:
        raise ValueError("Tập validation rỗng")

    vocab_size = len(vectorizer.get_feature_names_out())
    matrix = transform_tfidf_pipeline(vectorizer, text_list)

    total_cells = matrix.shape[0] * matrix.shape[1]
    sparsity_pct = float((1.0 - (matrix.nnz / total_cells)) * 100.0) if total_cells > 0 else 0.0

    doc_nnz = np.diff(matrix.indptr)
    zero_vector_docs = int(np.sum(doc_nnz == 0))
    zero_vector_pct = float((zero_vector_docs / num_docs) * 100.0)

    mean_active = float(np.mean(doc_nnz)) if num_docs > 0 else 0.0
    min_active = int(np.min(doc_nnz)) if num_docs > 0 else 0
    max_active = int(np.max(doc_nnz)) if num_docs > 0 else 0

    analyzer = vectorizer.build_analyzer()
    vocab_set = set(vectorizer.vocabulary_.keys())

    total_tokens = 0
    oov_tokens = 0

    for text in text_list:
        tokens = analyzer(text)
        total_tokens += len(tokens)
        for token in tokens:
            if token not in vocab_set:
                oov_tokens += 1

    oov_rate_pct = float((oov_tokens / total_tokens) * 100.0) if total_tokens > 0 else 0.0

    return {
        "vocab_size": vocab_size,
        "val_num_docs": num_docs,
        "val_sparsity_pct": round(sparsity_pct, 4),
        "val_mean_active_terms": round(mean_active, 2),
        "val_min_active_terms": min_active,
        "val_max_active_terms": max_active,
        "val_zero_vector_docs": zero_vector_docs,
        "val_zero_vector_pct": round(zero_vector_pct, 4),
        "val_total_tokens": total_tokens,
        "val_oov_tokens": oov_tokens,
        "val_oov_rate_pct": round(oov_rate_pct, 4),
    }


def save_tfidf_artifacts(
    output_dir: str | Path,
    vectorizer: TfidfVectorizer,
    config: dict[str, Any],
    run_metadata: dict[str, Any],
    splits_data: dict[str, tuple[csr_matrix, list[str]]],
) -> None:
    """Lưu trữ đầy đủ artifacts TF-IDF theo đúng hợp đồng Handbook Mục E1.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, out_path / "tfidf_vectorizer.joblib")

    feature_names = vectorizer.get_feature_names_out().tolist()
    with open(out_path / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    with open(out_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    with open(out_path / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, ensure_ascii=False, indent=2)

    for split_name, (matrix, article_ids) in splits_data.items():
        if not isinstance(matrix, csr_matrix):
            matrix = matrix.tocsr()

        if matrix.shape[0] != len(article_ids):
            raise ValueError(
                f"Split {split_name}: số dòng ma trận ({matrix.shape[0]}) "
                f"không khớp số article_id ({len(article_ids)})"
            )

        matrix_path = out_path / f"tfidf_{split_name}.npz"
        save_npz(matrix_path, matrix)

        ids_path = out_path / f"{split_name}_article_ids.json"
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump([str(aid) for aid in article_ids], f, ensure_ascii=False, indent=2)


def load_tfidf_artifacts(
    artifact_dir: str | Path,
    split: str = "train",
) -> dict[str, Any]:
    """Tải artifacts TF-IDF cho một split và xác thực tính toàn vẹn."""
    dir_path = Path(artifact_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục artifacts: {dir_path}")

    vec_path = dir_path / "tfidf_vectorizer.joblib"
    if not vec_path.exists():
        raise FileNotFoundError(f"Không tìm thấy vectorizer: {vec_path}")
    vectorizer: TfidfVectorizer = joblib.load(vec_path)

    features_path = dir_path / "feature_names.json"
    with open(features_path, "r", encoding="utf-8") as f:
        feature_names: list[str] = json.load(f)

    cfg_path = dir_path / "config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = json.load(f)

    meta_path = dir_path / "run_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        run_metadata: dict[str, Any] = json.load(f)

    matrix_path = dir_path / f"tfidf_{split}.npz"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Không tìm thấy ma trận cho split {split}: {matrix_path}")
    matrix = load_npz(matrix_path)

    ids_path = dir_path / f"{split}_article_ids.json"
    if not ids_path.exists():
        raise FileNotFoundError(f"Không tìm thấy danh sách article_ids cho split {split}: {ids_path}")
    with open(ids_path, "r", encoding="utf-8") as f:
        article_ids: list[str] = json.load(f)

    if matrix.shape[0] != len(article_ids):
        raise ValueError(
            f"Kích thước không khớp: {matrix.shape[0]} hàng trong ma trận "
            f"nhưng có {len(article_ids)} article_ids"
        )
    if matrix.shape[1] != len(feature_names):
        raise ValueError(
            f"Số cột không khớp: {matrix.shape[1]} cột trong ma trận "
            f"nhưng có {len(feature_names)} feature names"
        )

    return {
        "matrix": matrix,
        "article_ids": article_ids,
        "vectorizer": vectorizer,
        "feature_names": feature_names,
        "config": config,
        "run_metadata": run_metadata,
    }

