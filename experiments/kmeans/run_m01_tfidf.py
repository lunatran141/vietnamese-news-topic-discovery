"""Thực thi pipeline trích xuất đặc trưng TF-IDF cho Task M01.

Script thực hiện:
1. Đọc dữ liệu phân chia Train/Val/Test đã chuẩn hóa.
2. Chạy grid search trên tập Val để đánh giá độ phủ và độ thưa của từ vựng.
3. Tự động chọn cấu hình tối ưu theo 4 tiêu chí định lượng (OOV thấp nhất, active terms hợp lý, không có zero vector, đúng budget).
4. Huấn luyện vectorizer duy nhất trên tập Train (Zero Leakage).
5. Biến đổi ma trận cho cả 3 tập Train, Val, Test.
6. Lưu trữ đầy đủ bộ artifacts theo hợp đồng Handbook Mục E1.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys
import time
from typing import Any

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")


import numpy as np
import pandas as pd
from scipy.sparse import issparse
import yaml

from src.data.dataset_loader import load_split_dataset
from src.features.article_text import build_article_text
from src.features.tfidf import (
    create_tfidf_vectorizer,
    evaluate_validation_vocabulary,
    fit_tfidf_pipeline,
    load_stopwords,
    save_tfidf_artifacts,
    transform_tfidf_pipeline,
)


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Chạy TF-IDF Feature Extraction Pipeline cho Task M01"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tfidf.yaml",
        help="Đường dẫn tới file cấu hình YAML (mặc định: configs/tfidf.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Đường dẫn thư mục lưu artifacts (ghi đè cấu hình nếu chỉ định)",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Bỏ qua bước grid search, dùng trực tiếp cấu hình mặc định trong YAML",
    )
    return parser.parse_args()


def run_grid_search(
    train_texts: list[str],
    val_texts: list[str],
    base_config: dict[str, Any],
    results_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Thực hiện grid search trên tập validation và chọn cấu hình tốt nhất."""
    tuning_grid = base_config.get("tuning_grid", {})
    if not tuning_grid:
        print("Không tìm thấy tuning_grid trong config. Dùng cấu hình mặc định.")
        return base_config.get("vectorizer", {}), pd.DataFrame()

    keys = list(tuning_grid.keys())
    grid_values = [tuning_grid[k] for k in keys]
    combinations = list(itertools.product(*grid_values))

    print(f"Bắt đầu tinh chỉnh tham số: {len(combinations)} tổ hợp cấu hình...")

    stopwords_file = base_config.get("vectorizer", {}).get("stopwords_file")
    stop_words = load_stopwords(stopwords_file) if stopwords_file else None

    # Tối ưu hóa: Tiền trích xuất n-grams của validation set theo từng ngram_range
    unique_ngram_ranges = list({tuple(c[keys.index("ngram_range")]) for c in combinations})
    val_token_counters: dict[tuple[int, int], Counter[str]] = {}

    for nr in unique_ngram_ranges:
        from sklearn.feature_extraction.text import TfidfVectorizer
        temp_vec = TfidfVectorizer(ngram_range=nr, stop_words=stop_words)
        analyzer = temp_vec.build_analyzer()
        counter: Counter[str] = Counter()
        for t in val_texts:
            counter.update(analyzer(t))
        val_token_counters[nr] = counter

    records: list[dict[str, Any]] = []

    for idx, combo in enumerate(combinations, start=1):
        candidate_params = dict(zip(keys, combo))
        nr = tuple(candidate_params["ngram_range"])

        # Tạo vectorizer candidate
        cand_cfg = {
            "ngram_range": nr,
            "min_df": candidate_params.get("min_df", 3),
            "max_df": candidate_params.get("max_df", 0.85),
            "max_features": candidate_params.get("max_features", 15000),
            "sublinear_tf": candidate_params.get("sublinear_tf", True),
            "norm": base_config.get("vectorizer", {}).get("norm", "l2"),
            "token_pattern": base_config.get("vectorizer", {}).get("token_pattern", r"(?u)\b\w\w+\b"),
            "stopwords_file": stopwords_file,
        }

        # Fit trên train_texts
        vectorizer = fit_tfidf_pipeline(train_texts, cand_cfg)
        vocab_size = len(vectorizer.get_feature_names_out())
        vocab_set = set(vectorizer.vocabulary_.keys())

        # Transform validation set
        val_matrix = transform_tfidf_pipeline(vectorizer, val_texts)
        doc_nnz = np.diff(val_matrix.indptr)
        zero_docs = int(np.sum(doc_nnz == 0))
        mean_active = float(np.mean(doc_nnz))
        min_active = int(np.min(doc_nnz))
        max_active = int(np.max(doc_nnz))

        total_cells = val_matrix.shape[0] * val_matrix.shape[1]
        sparsity_pct = float((1.0 - (val_matrix.nnz / total_cells)) * 100.0) if total_cells > 0 else 0.0

        # Tính OOV chính xác từ val_token_counters
        token_counter = val_token_counters[nr]
        total_tokens = sum(token_counter.values())
        oov_tokens = sum(cnt for tok, cnt in token_counter.items() if tok not in vocab_set)
        oov_rate_pct = float((oov_tokens / total_tokens) * 100.0) if total_tokens > 0 else 0.0

        # Điểm heuristic tổng hợp:
        # 1. zero vector docs bị phạt nặng
        # 2. oov_rate_pct càng thấp càng tốt
        # 3. mean_active càng gần vùng lý tưởng (150 từ kích hoạt) càng tốt
        penalty_zero = 10000.0 if zero_docs > 0 else 0.0
        active_term_dist = abs(mean_active - 150.0)
        heuristic_score = oov_rate_pct + (active_term_dist * 0.05) + penalty_zero

        record = {
            "candidate_id": idx,
            "ngram_range": f"{nr[0]}-{nr[1]}",
            "min_df": candidate_params.get("min_df"),
            "max_df": candidate_params.get("max_df"),
            "max_features": candidate_params.get("max_features"),
            "sublinear_tf": candidate_params.get("sublinear_tf"),
            "vocab_size": vocab_size,
            "val_oov_rate_pct": round(oov_rate_pct, 4),
            "val_mean_active_terms": round(mean_active, 2),
            "val_min_active_terms": min_active,
            "val_max_active_terms": max_active,
            "val_zero_vector_docs": zero_docs,
            "val_zero_vector_pct": round((zero_docs / len(val_texts)) * 100.0, 4),
            "val_sparsity_pct": round(sparsity_pct, 4),
            "heuristic_score": round(heuristic_score, 4),
            "params_dict": cand_cfg,
        }
        records.append(record)

    results_df = pd.DataFrame(records)

    # Chọn cấu hình có heuristic_score thấp nhất
    best_idx = results_df["heuristic_score"].idxmin()
    results_df["is_best"] = False
    results_df.loc[best_idx, "is_best"] = True

    best_record = results_df.loc[best_idx]
    best_params: dict[str, Any] = best_record["params_dict"]

    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "tfidf_tuning_validation_results.csv"
    export_df = results_df.drop(columns=["params_dict"])
    export_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Đã lưu kết quả tuning vào: {csv_path}")

    print(
        f"\nCấu hình tối ưu được chọn (Candidate {best_record['candidate_id']}):"
        f"\n  - ngram_range: {best_record['ngram_range']}"
        f"\n  - min_df: {best_record['min_df']}"
        f"\n  - max_df: {best_record['max_df']}"
        f"\n  - max_features: {best_record['max_features']}"
        f"\n  - val_oov_rate_pct: {best_record['val_oov_rate_pct']}%"
        f"\n  - val_mean_active_terms: {best_record['val_mean_active_terms']}"
        f"\n  - val_zero_vector_docs: {best_record['val_zero_vector_docs']}"
        f"\n  - val_sparsity_pct: {best_record['val_sparsity_pct']}%"
    )

    return best_params, results_df


def main() -> None:
    """Hàm chính điều phối pipeline M01."""
    start_time = time.time()
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy cấu hình: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    dataset_cfg = config["dataset"]
    output_dir = Path(args.output_dir or config.get("artifacts", {}).get("output_dir", "artifacts/tfidf"))
    results_dir = Path("results/tfidf")

    print(f"Đang đọc dữ liệu theo chuẩn Dataset v1: {dataset_cfg['path']}")
    train_df = load_split_dataset(
        dataset_cfg["path"],
        dataset_cfg["split_manifest_path"],
        dataset_cfg["train_split"],
    )
    val_df = load_split_dataset(
        dataset_cfg["path"],
        dataset_cfg["split_manifest_path"],
        dataset_cfg["val_split"],
    )
    test_df = load_split_dataset(
        dataset_cfg["path"],
        dataset_cfg["split_manifest_path"],
        dataset_cfg["test_split"],
    )

    text_variant = dataset_cfg.get("text_variant", "text")
    print(f"Trích xuất văn bản đầu vào biến thể: '{text_variant}'")
    train_texts = build_article_text(train_df, text_variant).tolist()
    val_texts = build_article_text(val_df, text_variant).tolist()
    test_texts = build_article_text(test_df, text_variant).tolist()

    train_ids = train_df["article_id"].astype(str).tolist()
    val_ids = val_df["article_id"].astype(str).tolist()
    test_ids = test_df["article_id"].astype(str).tolist()

    print(f"Số lượng bài viết: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")

    # Bước Tuning hoặc dùng mặc định
    if args.skip_tuning:
        print("Bỏ qua tuning, sử dụng cấu hình mặc định từ YAML.")
        chosen_vec_cfg = config.get("vectorizer", {})
        tuning_summary = {"status": "skipped"}
    else:
        chosen_vec_cfg, tuning_df = run_grid_search(
            train_texts,
            val_texts,
            config,
            results_dir,
        )
        tuning_summary = {
            "status": "completed",
            "total_candidates": len(tuning_df),
            "best_candidate_id": int(tuning_df.loc[tuning_df["is_best"], "candidate_id"].iloc[0]),
            "best_val_oov_rate_pct": float(tuning_df.loc[tuning_df["is_best"], "val_oov_rate_pct"].iloc[0]),
            "best_val_mean_active": float(tuning_df.loc[tuning_df["is_best"], "val_mean_active_terms"].iloc[0]),
        }

    # Bước Huấn luyện chính thức: FIT DUY NHẤT TRÊN TRAIN
    print("\nHuấn luyện TfidfVectorizer duy nhất trên tập Train (Zero Leakage)...")
    final_config_dict = {
        "dataset": dataset_cfg,
        "vectorizer": chosen_vec_cfg,
        "selected_at": datetime.now(timezone.utc).isoformat(),
    }

    final_vectorizer = fit_tfidf_pipeline(train_texts, chosen_vec_cfg)
    feature_names = final_vectorizer.get_feature_names_out().tolist()
    vocab_size = len(feature_names)
    print(f"Kích thước từ vựng đã học: {vocab_size} từ/cụm từ")

    # Transform cả 3 split
    print("Biến đổi ma trận thưa cho Train, Val và Test...")
    train_matrix = transform_tfidf_pipeline(final_vectorizer, train_texts)
    val_matrix = transform_tfidf_pipeline(final_vectorizer, val_texts)
    test_matrix = transform_tfidf_pipeline(final_vectorizer, test_texts)

    # Đánh giá tập validation
    val_eval = evaluate_validation_vocabulary(final_vectorizer, val_texts, val_ids)

    # Tính độ thưa của cả 3 tập
    train_cells = train_matrix.shape[0] * train_matrix.shape[1]
    train_sparsity = float((1.0 - (train_matrix.nnz / train_cells)) * 100.0)

    val_cells = val_matrix.shape[0] * val_matrix.shape[1]
    val_sparsity = float((1.0 - (val_matrix.nnz / val_cells)) * 100.0)

    test_cells = test_matrix.shape[0] * test_matrix.shape[1]
    test_sparsity = float((1.0 - (test_matrix.nnz / test_cells)) * 100.0)

    # Kiểm tra tính toàn vẹn (finite, non-empty)
    for name, mat in [("Train", train_matrix), ("Val", val_matrix), ("Test", test_matrix)]:
        if not np.isfinite(mat.data).all():
            raise ValueError(f"Ma trận {name} chứa giá trị NaN hoặc Vô cực!")
        if mat.shape[0] == 0 or mat.shape[1] == 0:
            raise ValueError(f"Ma trận {name} rỗng!")

    duration = round(time.time() - start_time, 2)

    # Tạo run_metadata.json
    run_metadata = {
        "pipeline_name": "M01_tfidf_pipeline",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_duration_seconds": duration,
        "input_dataset": dataset_cfg["path"],
        "data_version": dataset_cfg.get("data_version", "v1.0"),
        "preprocessing_variant": dataset_cfg.get("preprocessing_variant", "p2_segmented"),
        "text_variant": text_variant,
        "vocab_size": vocab_size,
        "shapes": {
            "train": [int(train_matrix.shape[0]), int(train_matrix.shape[1])],
            "val": [int(val_matrix.shape[0]), int(val_matrix.shape[1])],
            "test": [int(test_matrix.shape[0]), int(test_matrix.shape[1])],
        },
        "non_zeros": {
            "train": int(train_matrix.nnz),
            "val": int(val_matrix.nnz),
            "test": int(test_matrix.nnz),
        },
        "sparsity_percentages": {
            "train": round(train_sparsity, 4),
            "val": round(val_sparsity, 4),
            "test": round(test_sparsity, 4),
        },
        "validation_evaluation": val_eval,
        "tuning_summary": tuning_summary,
    }

    # Lưu trữ toàn bộ 10 artifacts theo Mục E1 Handbook
    print(f"\nLưu trữ artifacts vào: {output_dir}")
    splits_data = {
        "train": (train_matrix, train_ids),
        "val": (val_matrix, val_ids),
        "test": (test_matrix, test_ids),
    }

    save_tfidf_artifacts(
        output_dir=output_dir,
        vectorizer=final_vectorizer,
        config=final_config_dict,
        run_metadata=run_metadata,
        splits_data=splits_data,
    )

    # In top IDF terms
    idf_values = final_vectorizer.idf_
    sorted_idf_indices = np.argsort(idf_values)
    lowest_idf_terms = [(feature_names[i], round(float(idf_values[i]), 4)) for i in sorted_idf_indices[:15]]
    highest_idf_terms = [(feature_names[i], round(float(idf_values[i]), 4)) for i in sorted_idf_indices[-15:]]

    print("\nHoàn tất pipeline M01 thành công!")
    print(f"Tổng thời gian: {duration} giây")
    print(f"Train Shape: {train_matrix.shape} (Độ thưa: {train_sparsity:.2f}%)")
    print(f"Val Shape:   {val_matrix.shape} (Độ thưa: {val_sparsity:.2f}%, OOV: {val_eval['val_oov_rate_pct']}%)")
    print(f"Test Shape:  {test_matrix.shape} (Độ thưa: {test_sparsity:.2f}%)")
    print(f"Active terms trung bình trên Val: {val_eval['val_mean_active_terms']} từ/bài")
    print(f"Số bài rỗng trên Val: {val_eval['val_zero_vector_docs']}")
    print("\nTop 15 từ có IDF thấp nhất (xuất hiện phổ biến nhất):")
    for term, val in lowest_idf_terms:
        print(f"  - {term}: {val}")
    print("\nTop 15 từ có IDF cao nhất (từ khóa đặc thù nhất):")
    for term, val in highest_idf_terms:
        print(f"  - {term}: {val}")


if __name__ == "__main__":
    main()
