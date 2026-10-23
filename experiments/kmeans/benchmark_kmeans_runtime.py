"""Benchmark thời gian thực thi và tài nguyên phần cứng cho mô hình TF-IDF + K-Means (Task E08).
Tuân thủ nghiêm ngặt quy định tại Mục 7 trong docs/evaluation_protocol.md.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import sys
import time
import tracemalloc
from typing import Any

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

import numpy as np
import pandas as pd
import psutil

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_loader import load_split_dataset
from src.features.tfidf import fit_tfidf_pipeline, transform_tfidf_pipeline
from src.models.kmeans import compute_clustering_metrics, predict_kmeans, train_kmeans


def get_hardware_info() -> dict[str, Any]:
    """Lấy thông số phần cứng của môi trường thực thi."""
    cpu_name = platform.processor() or "Unknown CPU"
    cpu_count_logical = psutil.cpu_count(logical=True)
    cpu_count_physical = psutil.cpu_count(logical=False)
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    os_name = f"{platform.system()} {platform.release()} ({platform.machine()})"
    python_ver = platform.python_version()

    return {
        "cpu_name": cpu_name,
        "cpu_cores_physical": cpu_count_physical,
        "cpu_cores_logical": cpu_count_logical,
        "ram_total_gb": ram_gb,
        "os": os_name,
        "python_version": python_ver,
        "summary": f"{cpu_count_logical} vCPUs, {ram_gb} GB RAM, {platform.system()}",
    }


def run_single_benchmark_pass(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 9,
    seed: int = 42,
) -> dict[str, float]:
    """Chạy 1 lần đo pipeline toàn trình và đo thời gian từng giai đoạn bằng time.perf_counter()."""
    tracemalloc.start()
    
    # 1. Giai đoạn Biểu diễn đặc trưng (representation)
    t0 = time.perf_counter()
    tfidf_cfg = {
        "ngram_range": (1, 1),
        "min_df": 2,
        "max_df": 0.85,
        "max_features": 20000,
        "sublinear_tf": True,
        "norm": "l2",
    }
    vectorizer = fit_tfidf_pipeline(train_df["text"].tolist(), tfidf_cfg)
    train_matrix = transform_tfidf_pipeline(vectorizer, train_df["text"].tolist())
    val_matrix = transform_tfidf_pipeline(vectorizer, val_df["text"].tolist())
    test_matrix = transform_tfidf_pipeline(vectorizer, test_df["text"].tolist())
    t1 = time.perf_counter()
    representation_sec = t1 - t0

    # 2. Giai đoạn Huấn luyện (fit)
    t2 = time.perf_counter()
    model = train_kmeans(
        train_matrix,
        n_clusters=k,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=seed,
    )
    t3 = time.perf_counter()
    fit_sec = t3 - t2

    # 3. Giai đoạn Suy luận (predict)
    t4 = time.perf_counter()
    val_preds = predict_kmeans(model, val_matrix)
    test_preds = predict_kmeans(model, test_matrix)
    t5 = time.perf_counter()
    predict_sec = t5 - t4

    # 4. Giai đoạn Đánh giá (evaluation)
    t6 = time.perf_counter()
    test_gold = test_df["category_gold"].tolist()
    _ = compute_clustering_metrics(
        test_matrix,
        test_preds,
        model=model,
        gold_labels=test_gold,
    )
    t7 = time.perf_counter()
    eval_sec = t7 - t6

    # RAM đỉnh
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram_mb = peak / (1024 * 1024)

    total_pipeline_sec = representation_sec + fit_sec + predict_sec + eval_sec

    return {
        "representation_seconds": representation_sec,
        "fit_seconds": fit_sec,
        "predict_seconds": predict_sec,
        "evaluation_seconds": eval_sec,
        "total_pipeline_seconds": total_pipeline_sec,
        "peak_ram_mb": peak_ram_mb,
    }


def measure_model_size_mb() -> float:
    """Tính dung lượng lưu trữ các file mô hình K-Means và TF-IDF artifacts trên đĩa."""
    target_files = [
        PROJECT_ROOT / "artifacts" / "tfidf" / "tfidf_vectorizer.joblib",
        PROJECT_ROOT / "artifacts" / "kmeans" / "kmeans_model.joblib",
    ]
    total_bytes = 0
    for p in target_files:
        if p.is_file():
            total_bytes += p.stat().st_size
    return round(total_bytes / (1024 * 1024), 2)


def main() -> None:
    print("=" * 60)
    print("Bắt đầu Benchmark E08 cho TF-IDF + K-Means...")
    print("=" * 60)

    hw = get_hardware_info()
    print(f"Cấu hình máy: {hw['summary']}")
    print(f"Python: {hw['python_version']}")

    # Load dữ liệu P3 (chuẩn cho TF-IDF + K-Means)
    data_path = PROJECT_ROOT / "data" / "processed" / "p3_no_stopwords.csv"
    manifest_path = PROJECT_ROOT / "data" / "splits" / "split_manifest.csv"
    
    print("\nNạp dữ liệu Train, Validation, Test...")
    train_df = load_split_dataset(data_path, manifest_path, "train")
    val_df = load_split_dataset(data_path, manifest_path, "val")
    test_df = load_split_dataset(data_path, manifest_path, "test")
    print(f"Số lượng: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    n_repeats = 3
    print(f"\nTiến hành đo lặp lại {n_repeats} lần...")
    runs = []
    for i in range(n_repeats):
        print(f"  -> Lần chạy {i + 1}/{n_repeats}...", end=" ", flush=True)
        res = run_single_benchmark_pass(train_df, val_df, test_df, k=9, seed=42)
        runs.append(res)
        print(f"Xong trong {res['total_pipeline_seconds']:.2f}s (RAM đỉnh: {res['peak_ram_mb']:.1f} MB)")

    # Tổng hợp thống kê
    keys = list(runs[0].keys())
    stats = {}
    for k in keys:
        vals = [r[k] for r in runs]
        stats[f"{k}_mean"] = float(np.mean(vals))
        stats[f"{k}_std"] = float(np.std(vals))

    model_size_mb = measure_model_size_mb()

    benchmark_record = {
        "method": "KMeans (TF-IDF)",
        "run_id": "kmeans_baseline",
        "fit_split": "train",
        "n_train": len(train_df),
        "n_eval": len(test_df),
        "n_repeats": n_repeats,
        "representation_seconds": round(stats["representation_seconds_mean"], 3),
        "representation_seconds_std": round(stats["representation_seconds_std"], 3),
        "fit_seconds": round(stats["fit_seconds_mean"], 3),
        "fit_seconds_std": round(stats["fit_seconds_std"], 3),
        "predict_seconds": round(stats["predict_seconds_mean"], 3),
        "predict_seconds_std": round(stats["predict_seconds_std"], 3),
        "evaluation_seconds": round(stats["evaluation_seconds_mean"], 3),
        "evaluation_seconds_std": round(stats["evaluation_seconds_std"], 3),
        "total_pipeline_seconds": round(stats["total_pipeline_seconds_mean"], 3),
        "total_pipeline_seconds_std": round(stats["total_pipeline_seconds_std"], 3),
        "setup_seconds": 0.0,
        "ram_peak_mb": round(stats["peak_ram_mb_mean"], 1),
        "model_size_mb": model_size_mb,
        "hardware": hw["summary"],
        "python_version": hw["python_version"],
        "scope_notes": "TF-IDF fit Train + transform Val/Test + KMeans fit Train + predict Val/Test + eval Test; excludes disk I/O & plotting",
        "raw_runs": runs,
    }

    output_dir = PROJECT_ROOT / "results" / "runtime_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / "kmeans_runtime_benchmark.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_record, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("KẾT QUẢ BENCHMARK K-MEANS:")
    print(f"  - Representation (TF-IDF) : {benchmark_record['representation_seconds']}s ± {benchmark_record['representation_seconds_std']}s")
    print(f"  - Fit (K-Means K=9)       : {benchmark_record['fit_seconds']}s ± {benchmark_record['fit_seconds_std']}s")
    print(f"  - Predict (Val + Test)    : {benchmark_record['predict_seconds']}s ± {benchmark_record['predict_seconds_std']}s")
    print(f"  - Evaluation (Metrics)    : {benchmark_record['evaluation_seconds']}s ± {benchmark_record['evaluation_seconds_std']}s")
    print(f"  - Total Pipeline          : {benchmark_record['total_pipeline_seconds']}s ± {benchmark_record['total_pipeline_seconds_std']}s")
    print(f"  - Peak RAM                : {benchmark_record['ram_peak_mb']} MB")
    print(f"  - Model Size on Disk      : {benchmark_record['model_size_mb']} MB")
    print(f"  - Đã lưu tại: {out_json}")
    print("=" * 60)


if __name__ == "__main__":
    main()
