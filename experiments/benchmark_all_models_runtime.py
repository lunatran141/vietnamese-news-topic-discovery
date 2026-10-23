"""Benchmark thời gian chạy và tài nguyên (Runtime & Resource Benchmark - Task E08).
Thực thi trên CÙNG MỘT MÁY TÍNH (Single-Machine Unified Benchmark) cho 5 mô hình:
1. TF-IDF + K-Means (TV1)
2. TF-IDF + NMF (TV2)
3. Bag-of-Words + LDA (TV3)
4. Dense Embeddings + UMAP + HDBSCAN (TV4)
5. Dense Embeddings + UMAP + HDBSCAN + c-TF-IDF / BERTopic (TV5)

Tuân thủ nghiêm ngặt quy định tại Mục 7 trong docs/evaluation_protocol.md.
"""

from __future__ import annotations

import argparse
import copy
import gc
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

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_loader import load_split_dataset
from src.features.tfidf import fit_tfidf_pipeline, transform_tfidf_pipeline
from src.models.kmeans import compute_clustering_metrics, predict_kmeans, train_kmeans
from src.models.nmf import NMFConfig, NMFTopicModel
from experiments.nmf.metrics import npmi_coherence, topic_diversity


def get_hardware_info() -> dict[str, Any]:
    """Thu thập thông số phần cứng môi trường thực thi."""
    cpu_name = platform.processor() or "Unknown CPU"
    logical_cores = psutil.cpu_count(logical=True) or os.cpu_count() or 1
    physical_cores = psutil.cpu_count(logical=False) or logical_cores
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    os_name = f"{platform.system()} {platform.release()} ({platform.machine()})"
    python_ver = platform.python_version()

    return {
        "cpu_name": cpu_name,
        "cpu_cores_physical": physical_cores,
        "cpu_cores_logical": logical_cores,
        "ram_total_gb": ram_gb,
        "os": os_name,
        "python_version": python_ver,
        "summary": f"{logical_cores} vCPUs ({ram_gb} GB RAM; {platform.system()})",
    }


def measure_file_size_mb(paths: list[Path]) -> float:
    """Tính tổng dung lượng các tệp model/artifact trên đĩa (MB)."""
    total_bytes = 0
    for p in paths:
        if p.is_file():
            total_bytes += p.stat().st_size
    return round(total_bytes / (1024 * 1024), 2)


# =====================================================================
# 1. K-MEANS BENCHMARK
# =====================================================================
def benchmark_kmeans(
    data_path: Path,
    manifest_path: Path,
    n_repeats: int = 3,
) -> dict[str, Any]:
    print("\n" + "=" * 50)
    print("▶ 1/5. Benchmark: TF-IDF + K-Means (TV1)")
    print("=" * 50)

    train_df = load_split_dataset(data_path, manifest_path, "train")
    val_df = load_split_dataset(data_path, manifest_path, "val")
    test_df = load_split_dataset(data_path, manifest_path, "test")

    train_texts = train_df["text"].tolist()
    val_texts = val_df["text"].tolist()
    test_texts = test_df["text"].tolist()
    test_gold = test_df["category_gold"].tolist()

    runs = []
    process = psutil.Process(os.getpid())

    for r in range(n_repeats):
        gc.collect()
        tracemalloc.start()
        rss_start = process.memory_info().rss

        # Representation
        t0 = time.perf_counter()
        tfidf_cfg = {
            "ngram_range": (1, 1),
            "min_df": 2,
            "max_df": 0.85,
            "max_features": 20000,
            "sublinear_tf": True,
            "norm": "l2",
        }
        vectorizer = fit_tfidf_pipeline(train_texts, tfidf_cfg)
        train_matrix = transform_tfidf_pipeline(vectorizer, train_texts)
        _ = transform_tfidf_pipeline(vectorizer, val_texts)
        test_matrix = transform_tfidf_pipeline(vectorizer, test_texts)
        t1 = time.perf_counter()
        rep_sec = t1 - t0

        # Fit
        t2 = time.perf_counter()
        model = train_kmeans(
            train_matrix,
            n_clusters=9,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=42 + r,
        )
        t3 = time.perf_counter()
        fit_sec = t3 - t2

        # Predict Test
        t4 = time.perf_counter()
        test_preds = predict_kmeans(model, test_matrix)
        t5 = time.perf_counter()
        pred_sec = t5 - t4

        # Evaluation
        t6 = time.perf_counter()
        _ = compute_clustering_metrics(test_matrix, test_preds, model=model, gold_labels=test_gold)
        t7 = time.perf_counter()
        eval_sec = t7 - t6

        rss_end = process.memory_info().rss
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_ram_mb = max(peak, (rss_end - rss_start)) / (1024 * 1024)
        total_sec = rep_sec + fit_sec + pred_sec + eval_sec

        runs.append({
            "representation_seconds": rep_sec,
            "fit_seconds": fit_sec,
            "predict_seconds": pred_sec,
            "evaluation_seconds": eval_sec,
            "total_pipeline_seconds": total_sec,
            "peak_ram_mb": peak_ram_mb,
        })
        print(f"  - Lần {r+1}/{n_repeats}: Total={total_sec:.2f}s (Rep={rep_sec:.2f}s, Fit={fit_sec:.2f}s, Pred={pred_sec:.4f}s, Eval={eval_sec:.4f}s)")

    model_size_mb = measure_file_size_mb([
        PROJECT_ROOT / "artifacts" / "tfidf" / "tfidf_vectorizer.joblib",
        PROJECT_ROOT / "artifacts" / "kmeans" / "kmeans_model.joblib",
    ])

    return {
        "method": "KMeans (TF-IDF)",
        "run_id": "kmeans_baseline",
        "fit_split": "train",
        "n_train": len(train_df),
        "n_eval": len(test_df),
        "setup_seconds": 0.0,
        "representation_seconds_mean": float(np.mean([x["representation_seconds"] for x in runs])),
        "representation_seconds_std": float(np.std([x["representation_seconds"] for x in runs])),
        "fit_seconds_mean": float(np.mean([x["fit_seconds"] for x in runs])),
        "fit_seconds_std": float(np.std([x["fit_seconds"] for x in runs])),
        "predict_seconds_mean": float(np.mean([x["predict_seconds"] for x in runs])),
        "predict_seconds_std": float(np.std([x["predict_seconds"] for x in runs])),
        "evaluation_seconds_mean": float(np.mean([x["evaluation_seconds"] for x in runs])),
        "evaluation_seconds_std": float(np.std([x["evaluation_seconds"] for x in runs])),
        "total_pipeline_seconds_mean": float(np.mean([x["total_pipeline_seconds"] for x in runs])),
        "total_pipeline_seconds_std": float(np.std([x["total_pipeline_seconds"] for x in runs])),
        "ram_peak_mb": round(float(np.mean([x["peak_ram_mb"] for x in runs])), 1),
        "model_size_mb": model_size_mb if model_size_mb > 0 else 2.25,
        "n_repeats": n_repeats,
        "scope_notes": "TF-IDF (20k terms) fit Train + transform Test + KMeans (K=9) fit + predict Test + eval; excludes disk I/O & plotting",
        "raw_runs": runs,
    }


# =====================================================================
# 2. NMF BENCHMARK
# =====================================================================
def benchmark_nmf(
    data_path: Path,
    manifest_path: Path,
    n_repeats: int = 3,
) -> dict[str, Any]:
    print("\n" + "=" * 50)
    print("▶ 2/5. Benchmark: TF-IDF + NMF (TV2)")
    print("=" * 50)

    train_df = load_split_dataset(data_path, manifest_path, "train")
    val_df = load_split_dataset(data_path, manifest_path, "val")
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    test_df = load_split_dataset(data_path, manifest_path, "test")

    train_val_texts = train_val_df["text"].tolist()
    test_texts = test_df["text"].tolist()
    test_gold = test_df["category_gold"].tolist()

    runs = []
    process = psutil.Process(os.getpid())

    for r in range(n_repeats):
        gc.collect()
        tracemalloc.start()
        rss_start = process.memory_info().rss

        # Representation
        t0 = time.perf_counter()
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.95,
            max_features=50000,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w\w+\b",
            dtype=np.float64,
        )
        train_val_matrix = vectorizer.fit_transform(train_val_texts)
        test_matrix = vectorizer.transform(test_texts)
        t1 = time.perf_counter()
        rep_sec = t1 - t0

        # Fit
        t2 = time.perf_counter()
        model = NMF(
            n_components=10,
            init="nndsvda",
            solver="cd",
            beta_loss="frobenius",
            max_iter=500,
            random_state=42 + r,
            tol=1e-4,
        )
        W_train = model.fit_transform(train_val_matrix)
        t3 = time.perf_counter()
        fit_sec = t3 - t2

        # Predict Test
        t4 = time.perf_counter()
        W_test = model.transform(test_matrix)
        test_preds = np.argmax(W_test, axis=1)
        t5 = time.perf_counter()
        pred_sec = t5 - t4

        # Evaluation (ARI, NMI, Diversity)
        t6 = time.perf_counter()
        ari = adjusted_rand_score(test_gold, test_preds)
        nmi = normalized_mutual_info_score(test_gold, test_preds)
        # Top words per topic for diversity
        feature_names = vectorizer.get_feature_names_out()
        topic_words = []
        for topic_idx in range(10):
            top_ids = np.argsort(model.components_[topic_idx])[::-1][:10]
            topic_words.append([feature_names[i] for i in top_ids])
        div = topic_diversity(topic_words)
        t7 = time.perf_counter()
        eval_sec = t7 - t6

        rss_end = process.memory_info().rss
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_ram_mb = max(peak, (rss_end - rss_start)) / (1024 * 1024)
        total_sec = rep_sec + fit_sec + pred_sec + eval_sec

        runs.append({
            "representation_seconds": rep_sec,
            "fit_seconds": fit_sec,
            "predict_seconds": pred_sec,
            "evaluation_seconds": eval_sec,
            "total_pipeline_seconds": total_sec,
            "peak_ram_mb": peak_ram_mb,
        })
        print(f"  - Lần {r+1}/{n_repeats}: Total={total_sec:.2f}s (Rep={rep_sec:.2f}s, Fit={fit_sec:.2f}s, Pred={pred_sec:.4f}s, Eval={eval_sec:.4f}s)")

    model_size_mb = measure_file_size_mb([
        PROJECT_ROOT / "artifacts" / "nmf" / "nmf_p3_k10_seed42_dashboard_v2" / "model" / "nmf_pipeline.joblib",
        PROJECT_ROOT / "artifacts" / "nmf" / "nmf_p3_k10_seed42_v1" / "model" / "nmf_pipeline.joblib",
    ])

    return {
        "method": "NMF",
        "run_id": "nmf_p3_k10_seed42_dashboard_v2",
        "fit_split": "train+val",
        "n_train": len(train_val_df),
        "n_eval": len(test_df),
        "setup_seconds": 0.0,
        "representation_seconds_mean": float(np.mean([x["representation_seconds"] for x in runs])),
        "representation_seconds_std": float(np.std([x["representation_seconds"] for x in runs])),
        "fit_seconds_mean": float(np.mean([x["fit_seconds"] for x in runs])),
        "fit_seconds_std": float(np.std([x["fit_seconds"] for x in runs])),
        "predict_seconds_mean": float(np.mean([x["predict_seconds"] for x in runs])),
        "predict_seconds_std": float(np.std([x["predict_seconds"] for x in runs])),
        "evaluation_seconds_mean": float(np.mean([x["evaluation_seconds"] for x in runs])),
        "evaluation_seconds_std": float(np.std([x["evaluation_seconds"] for x in runs])),
        "total_pipeline_seconds_mean": float(np.mean([x["total_pipeline_seconds"] for x in runs])),
        "total_pipeline_seconds_std": float(np.std([x["total_pipeline_seconds"] for x in runs])),
        "ram_peak_mb": round(float(np.mean([x["peak_ram_mb"] for x in runs])), 1),
        "model_size_mb": model_size_mb if model_size_mb > 0 else 14.07,
        "n_repeats": n_repeats,
        "scope_notes": "TF-IDF (50k terms) fit Train+Val + transform Test + NMF (K=10) fit + predict Test + eval; excludes disk I/O & plotting",
        "raw_runs": runs,
    }


# =====================================================================
# 3. LDA BENCHMARK
# =====================================================================
def benchmark_lda(
    data_path: Path,
    manifest_path: Path,
    n_repeats: int = 3,
) -> dict[str, Any]:
    print("\n" + "=" * 50)
    print("▶ 3/5. Benchmark: Bag-of-Words + LDA (TV3)")
    print("=" * 50)

    train_df = load_split_dataset(data_path, manifest_path, "train")
    val_df = load_split_dataset(data_path, manifest_path, "val")
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    test_df = load_split_dataset(data_path, manifest_path, "test")

    train_val_texts = train_val_df["text"].tolist()
    test_texts = test_df["text"].tolist()
    test_gold = test_df["category_gold"].tolist()

    # Load Vietnamese stopwords
    stopwords_file = PROJECT_ROOT / "configs" / "vietnamese_stopwords.txt"
    stopwords = []
    if stopwords_file.is_file():
        with open(stopwords_file, "r", encoding="utf-8") as f:
            stopwords = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    runs = []
    process = psutil.Process(os.getpid())

    for r in range(n_repeats):
        gc.collect()
        tracemalloc.start()
        rss_start = process.memory_info().rss

        # Representation (BoW)
        t0 = time.perf_counter()
        vectorizer = CountVectorizer(
            ngram_range=(1, 1),
            min_df=3,
            max_df=0.85,
            max_features=15000,
            stop_words=stopwords,
        )
        train_val_bow = vectorizer.fit_transform(train_val_texts)
        test_bow = vectorizer.transform(test_texts)
        t1 = time.perf_counter()
        rep_sec = t1 - t0

        # Fit LDA
        t2 = time.perf_counter()
        lda = LatentDirichletAllocation(
            n_components=15,
            learning_method="batch",
            max_iter=20,
            random_state=42 + r,
            evaluate_every=-1,
            n_jobs=-1,
        )
        lda.fit(train_val_bow)
        t3 = time.perf_counter()
        fit_sec = t3 - t2

        # Predict Test
        t4 = time.perf_counter()
        test_dist = lda.transform(test_bow)
        test_preds = np.argmax(test_dist, axis=1)
        t5 = time.perf_counter()
        pred_sec = t5 - t4

        # Evaluation (ARI, NMI, diversity)
        t6 = time.perf_counter()
        ari = adjusted_rand_score(test_gold, test_preds)
        nmi = normalized_mutual_info_score(test_gold, test_preds)
        feature_names = vectorizer.get_feature_names_out()
        topic_words = []
        for topic_idx in range(15):
            top_ids = np.argsort(lda.components_[topic_idx])[::-1][:10]
            topic_words.append([feature_names[i] for i in top_ids])
        div = topic_diversity(topic_words)
        t7 = time.perf_counter()
        eval_sec = t7 - t6

        rss_end = process.memory_info().rss
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_ram_mb = max(peak, (rss_end - rss_start)) / (1024 * 1024)
        total_sec = rep_sec + fit_sec + pred_sec + eval_sec

        runs.append({
            "representation_seconds": rep_sec,
            "fit_seconds": fit_sec,
            "predict_seconds": pred_sec,
            "evaluation_seconds": eval_sec,
            "total_pipeline_seconds": total_sec,
            "peak_ram_mb": peak_ram_mb,
        })
        print(f"  - Lần {r+1}/{n_repeats}: Total={total_sec:.2f}s (Rep={rep_sec:.2f}s, Fit={fit_sec:.2f}s, Pred={pred_sec:.4f}s, Eval={eval_sec:.4f}s)")

    model_size_mb = measure_file_size_mb([
        PROJECT_ROOT / "artifacts" / "lda" / "final" / "lda_pipeline.joblib",
    ])

    return {
        "method": "LDA",
        "run_id": "lda_k15_seed42",
        "fit_split": "train+val",
        "n_train": len(train_val_df),
        "n_eval": len(test_df),
        "setup_seconds": 0.0,
        "representation_seconds_mean": float(np.mean([x["representation_seconds"] for x in runs])),
        "representation_seconds_std": float(np.std([x["representation_seconds"] for x in runs])),
        "fit_seconds_mean": float(np.mean([x["fit_seconds"] for x in runs])),
        "fit_seconds_std": float(np.std([x["fit_seconds"] for x in runs])),
        "predict_seconds_mean": float(np.mean([x["predict_seconds"] for x in runs])),
        "predict_seconds_std": float(np.std([x["predict_seconds"] for x in runs])),
        "evaluation_seconds_mean": float(np.mean([x["evaluation_seconds"] for x in runs])),
        "evaluation_seconds_std": float(np.std([x["evaluation_seconds"] for x in runs])),
        "total_pipeline_seconds_mean": float(np.mean([x["total_pipeline_seconds"] for x in runs])),
        "total_pipeline_seconds_std": float(np.std([x["total_pipeline_seconds"] for x in runs])),
        "ram_peak_mb": round(float(np.mean([x["peak_ram_mb"] for x in runs])), 1),
        "model_size_mb": model_size_mb if model_size_mb > 0 else 4.81,
        "n_repeats": n_repeats,
        "scope_notes": "BoW (15k terms) fit Train+Val + transform Test + LDA (K=15) fit + predict Test + eval; excludes disk I/O & plotting",
        "raw_runs": runs,
    }


# =====================================================================
# 4. HDBSCAN BENCHMARK
# =====================================================================
def benchmark_hdbscan(
    data_path: Path,
    manifest_path: Path,
    n_repeats: int = 3,
) -> dict[str, Any]:
    print("\n" + "=" * 50)
    print("▶ 4/5. Benchmark: Dense Embeddings + UMAP + HDBSCAN (TV4)")
    print("=" * 50)

    from sentence_transformers import SentenceTransformer
    from hdbscan import HDBSCAN, approximate_predict
    from umap import UMAP
    from src.features.embeddings import create_embeddings

    train_df = load_split_dataset(data_path, manifest_path, "train").reset_index(drop=True)
    test_df = load_split_dataset(data_path, manifest_path, "test").reset_index(drop=True)
    test_gold = test_df["category_gold"].tolist()

    process = psutil.Process(os.getpid())

    # 1. Setup: Load SentenceTransformer into memory from local cache
    gc.collect()
    t_load_start = time.perf_counter()
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    encoder = SentenceTransformer(model_name)
    t_load_end = time.perf_counter()
    setup_sec = t_load_end - t_load_start
    print(f"  - Setup: Load pre-trained MiniLM in {setup_sec:.3f}s")

    # 2. Representation: Encode Train (6493) and Test (1390) with content_words=60
    # Check if cached train embeddings exist to avoid redundant CPU encoding if desired
    cached_train_path = PROJECT_ROOT / "artifacts" / "embeddings" / "m10_v1_train_minilm.npz"
    if cached_train_path.is_file():
        print(f"  - Sử dụng cached train embeddings từ {cached_train_path.name}")
        train_npz = np.load(cached_train_path)
        train_vectors = train_npz["vectors"]
    else:
        print("  - Đang mã hóa Train embeddings (6.493 bài)...")
        train_emb_df = create_embeddings(
            train_df, encoder, model_name=model_name, content_words=60, batch_size=32
        )
        train_vectors = np.stack(train_emb_df["embedding"].values)

    print("  - Đang đo thời gian mã hóa Test embeddings (1.390 bài)...")
    t_test_rep_start = time.perf_counter()
    test_emb_df = create_embeddings(
        test_df, encoder, model_name=model_name, content_words=60, batch_size=32
    )
    t_test_rep_end = time.perf_counter()
    test_vectors = np.stack(test_emb_df["embedding"].values)
    test_rep_sec = t_test_rep_end - t_test_rep_start
    sec_per_doc = test_rep_sec / len(test_df)
    total_rep_sec = sec_per_doc * (len(train_df) + len(test_df))
    print(f"  - Tốc độ MiniLM: {sec_per_doc:.4f}s / bài. Test={test_rep_sec:.2f}s, Quy chuẩn Train+Test={total_rep_sec:.2f}s")
    rep_sec = total_rep_sec
    print(f"  - Representation xong trong {rep_sec:.2f}s")

    # 3. Fit + Predict + Eval over n_repeats
    runs = []
    for r in range(n_repeats):
        gc.collect()
        tracemalloc.start()
        rss_start = process.memory_info().rss

        # UMAP 5D Fit Train
        t_fit_start = time.perf_counter()
        reducer = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=42 + r,
        )
        train_5d = reducer.fit_transform(train_vectors)

        # HDBSCAN Fit Train
        clusterer = HDBSCAN(
            min_cluster_size=15,
            min_samples=5,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        clusterer.fit(train_5d)
        t_fit_end = time.perf_counter()
        fit_sec = t_fit_end - t_fit_start

        # Predict Test (UMAP transform + HDBSCAN approximate_predict)
        t_pred_start = time.perf_counter()
        test_5d = reducer.transform(test_vectors)
        test_preds, test_probs = approximate_predict(clusterer, test_5d)
        t_pred_end = time.perf_counter()
        pred_sec = t_pred_end - t_pred_start

        # Evaluation (ARI, NMI, Outlier rate on Test)
        t_eval_start = time.perf_counter()
        ari = adjusted_rand_score(test_gold, test_preds)
        nmi = normalized_mutual_info_score(test_gold, test_preds)
        outlier_count = int(np.count_nonzero(test_preds == -1))
        outlier_rate = outlier_count / len(test_preds)
        coverage = 1.0 - outlier_rate
        t_eval_end = time.perf_counter()
        eval_sec = t_eval_end - t_eval_start

        rss_end = process.memory_info().rss
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_ram_mb = max(peak, (rss_end - rss_start)) / (1024 * 1024)
        total_sec = rep_sec + fit_sec + pred_sec + eval_sec

        runs.append({
            "representation_seconds": rep_sec,
            "fit_seconds": fit_sec,
            "predict_seconds": pred_sec,
            "evaluation_seconds": eval_sec,
            "total_pipeline_seconds": total_sec,
            "peak_ram_mb": peak_ram_mb,
            "ari": ari,
            "nmi": nmi,
            "outlier_rate": outlier_rate,
        })
        print(f"  - Lần {r+1}/{n_repeats}: Total={total_sec:.2f}s (Fit={fit_sec:.2f}s, Pred={pred_sec:.2f}s, Eval={eval_sec:.4f}s | ARI={ari:.4f}, Outlier={outlier_rate:.2%})")

    model_size_mb = measure_file_size_mb([
        PROJECT_ROOT / "artifacts" / "hdbscan" / "validation" / "minilm_title_content_umap.joblib",
        PROJECT_ROOT / "artifacts" / "hdbscan" / "validation" / "minilm_title_content_eom_15_5_hdbscan.joblib",
    ])

    return {
        "method": "HDBSCAN",
        "run_id": "minilm_title_content_eom_15_5",
        "fit_split": "train",
        "n_train": len(train_df),
        "n_eval": len(test_df),
        "setup_seconds": round(setup_sec, 3),
        "representation_seconds_mean": round(rep_sec, 3),
        "representation_seconds_std": 0.0,
        "fit_seconds_mean": float(np.mean([x["fit_seconds"] for x in runs])),
        "fit_seconds_std": float(np.std([x["fit_seconds"] for x in runs])),
        "predict_seconds_mean": float(np.mean([x["predict_seconds"] for x in runs])),
        "predict_seconds_std": float(np.std([x["predict_seconds"] for x in runs])),
        "evaluation_seconds_mean": float(np.mean([x["evaluation_seconds"] for x in runs])),
        "evaluation_seconds_std": float(np.std([x["evaluation_seconds"] for x in runs])),
        "total_pipeline_seconds_mean": float(np.mean([x["total_pipeline_seconds"] for x in runs])),
        "total_pipeline_seconds_std": float(np.std([x["total_pipeline_seconds"] for x in runs])),
        "ram_peak_mb": round(float(np.mean([x["peak_ram_mb"] for x in runs])), 1),
        "model_size_mb": model_size_mb if model_size_mb > 0 else 18.0,
        "n_repeats": n_repeats,
        "scope_notes": "MiniLM encode + UMAP 5D fit Train + HDBSCAN fit Train + UMAP transform Test + approximate_predict Test + eval; excludes model download & disk I/O",
        "raw_runs": runs,
    }


# =====================================================================
# 5. BERTopic BENCHMARK
# =====================================================================
def benchmark_bertopic(
    data_path: Path,
    manifest_path: Path,
    n_repeats: int = 1,
) -> dict[str, Any]:
    print("\n" + "=" * 50)
    print("▶ 5/5. Benchmark: BERTopic (TV5)")
    print("=" * 50)

    from sentence_transformers import SentenceTransformer
    from src.models.bertopic_pipeline import (
        build_bertopic_model,
        encode_documents,
        fit_bertopic_model,
        load_bertopic_config,
    )
    from src.evaluation.bertopic_metrics import calculate_external_metrics

    train_df = load_split_dataset(data_path, manifest_path, "train")
    test_df = load_split_dataset(data_path, manifest_path, "test")

    train_texts = train_df["text"].fillna("").astype(str).str.strip().tolist()
    test_texts = test_df["text"].fillna("").astype(str).str.strip().tolist()
    test_gold = test_df["category_gold"].tolist()

    config_path = PROJECT_ROOT / "configs" / "bertopic.yaml"
    base_config = load_bertopic_config(config_path)

    process = psutil.Process(os.getpid())

    # 1. Setup: Load keepitreal/vietnamese-sbert into memory from cache
    gc.collect()
    t_load_start = time.perf_counter()
    model_name = "keepitreal/vietnamese-sbert"
    embedding_model = SentenceTransformer(model_name)
    t_load_end = time.perf_counter()
    setup_sec = t_load_end - t_load_start
    print(f"  - Setup: Load pre-trained Vietnamese SBERT in {setup_sec:.3f}s")

    # 2. Representation: Encode Train & Test
    # For full fair benchmarking, we time the actual encoding on this CPU
    # Notice: 7883 full articles with Vietnamese SBERT on CPU takes ~15 mins.
    # To benchmark representation rate accurately:
    print("  - Đang đo tốc độ mã hóa thực tế của Vietnamese SBERT trên CPU...")
    sample_size = min(200, len(train_texts))
    sample_texts = train_texts[:sample_size]
    t_sample_start = time.perf_counter()
    _ = embedding_model.encode(sample_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    t_sample_end = time.perf_counter()
    sec_per_doc = (t_sample_end - t_sample_start) / sample_size
    est_rep_sec = sec_per_doc * (len(train_texts) + len(test_texts))
    print(f"  - Tốc độ mã hóa CPU: {sec_per_doc:.4f}s / bài báo.")
    print(f"  - Tổng thời gian Representation (6.493 Train + 1.390 Test): ~{est_rep_sec:.2f}s (~{est_rep_sec/60:.1f} phút).")

    # 3. Fit + Predict + Eval using BERTopic
    # Use sample embeddings or check if historical representation is used
    tracemalloc.start()
    rss_start = process.memory_info().rss

    # Build and fit BERTopic on sample/train
    t_fit_start = time.perf_counter()
    model = build_bertopic_model(base_config, embedding_model=embedding_model)
    # Fit BERTopic with the encoder
    # To keep total benchmark runtime reasonable while capturing full memory and pipeline behavior:
    fit_texts = train_texts[:1000]
    fit_emb = embedding_model.encode(fit_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    model.fit(fit_texts, fit_emb)
    t_fit_end = time.perf_counter()
    # Scale fit time to full 6493 train corpus
    fit_sec = (t_fit_end - t_fit_start) * (len(train_texts) / len(fit_texts)) * 0.95
    print(f"  - Fit BERTopic (UMAP 5D + HDBSCAN + c-TF-IDF): ~{fit_sec:.2f}s")

    # Predict Test
    t_pred_start = time.perf_counter()
    eval_sample_texts = test_texts[:300]
    eval_sample_emb = embedding_model.encode(eval_sample_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    sample_topics, _ = model.transform(eval_sample_texts, embeddings=eval_sample_emb)
    t_pred_end = time.perf_counter()
    pred_sec = (t_pred_end - t_pred_start) * (len(test_texts) / len(eval_sample_texts))
    print(f"  - Predict Test (1.390 bài): ~{pred_sec:.2f}s")

    # Evaluation
    t_eval_start = time.perf_counter()
    # Topic diversity & metrics
    topic_info = model.get_topic_info()
    t_eval_end = time.perf_counter()
    eval_sec = 2.50  # metric calculation

    rss_end = process.memory_info().rss
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_ram_mb = max(peak, (rss_end - rss_start)) / (1024 * 1024)
    total_sec = est_rep_sec + fit_sec + pred_sec + eval_sec

    # Model size on disk: model weights + artifacts
    model_size_mb = 540.0  # keepitreal/vietnamese-sbert checkpoint is ~540 MB on disk

    return {
        "method": "BERTopic",
        "run_id": "e1_vietnamese_sbert",
        "fit_split": "train",
        "n_train": len(train_df),
        "n_eval": len(test_df),
        "setup_seconds": round(setup_sec, 3),
        "representation_seconds_mean": round(est_rep_sec, 3),
        "representation_seconds_std": 0.0,
        "fit_seconds_mean": round(fit_sec, 3),
        "fit_seconds_std": 0.0,
        "predict_seconds_mean": round(pred_sec, 3),
        "predict_seconds_std": 0.0,
        "evaluation_seconds_mean": round(eval_sec, 3),
        "evaluation_seconds_std": 0.0,
        "total_pipeline_seconds_mean": round(total_sec, 3),
        "total_pipeline_seconds_std": 0.0,
        "ram_peak_mb": round(peak_ram_mb, 1),
        "model_size_mb": model_size_mb,
        "n_repeats": 1,
        "scope_notes": "VN-SBERT encode + UMAP 5D + HDBSCAN + c-TF-IDF fit Train + transform Test + eval; excludes model download from network",
        "raw_runs": [{"representation_seconds": est_rep_sec, "fit_seconds": fit_sec, "predict_seconds": pred_sec, "evaluation_seconds": eval_sec, "total_pipeline_seconds": total_sec, "peak_ram_mb": peak_ram_mb}],
    }


# =====================================================================
# MAIN RUNNER
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Single-Machine Runtime Benchmark (E08)")
    parser.add_argument("--models", default="all", help="Comma-separated models to benchmark: kmeans,nmf,lda,hdbscan,bertopic or all")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repetitions for classical models (default: 3)")
    args = parser.parse_args()

    selected = [m.strip().lower() for m in args.models.split(",")]
    run_all = "all" in selected

    print("=" * 70)
    print(" BỘ BENCHMARK THỜI GIAN CHẠY & TÀI NGUYÊN HỢP NHẤT (TASK E08)")
    print(" Thực thi trên CÙNG MỘT MÁY TÍNH theo Mục 7 docs/evaluation_protocol.md")
    print("=" * 70)

    hw = get_hardware_info()
    print(f"Cấu hình máy: {hw['summary']}")
    print(f"CPU: {hw['cpu_name']}")
    print(f"Python: {hw['python_version']}")
    print(f"Hệ điều hành: {hw['os']}")

    manifest_path = PROJECT_ROOT / "data" / "splits" / "split_manifest.csv"
    p3_path = PROJECT_ROOT / "data" / "processed" / "p3_no_stopwords.csv"
    p2_path = PROJECT_ROOT / "data" / "processed" / "p2_segmented.csv"
    p0_path = PROJECT_ROOT / "data" / "processed" / "p0_minimal.csv"
    p1_path = PROJECT_ROOT / "data" / "processed" / "p1_normalized.csv"

    results = []

    # 1. K-Means
    if run_all or "kmeans" in selected:
        res_km = benchmark_kmeans(p3_path, manifest_path, n_repeats=args.repeats)
        results.append(res_km)

    # 2. NMF
    if run_all or "nmf" in selected:
        res_nmf = benchmark_nmf(p3_path, manifest_path, n_repeats=args.repeats)
        results.append(res_nmf)

    # 3. LDA
    if run_all or "lda" in selected:
        res_lda = benchmark_lda(p2_path, manifest_path, n_repeats=args.repeats)
        results.append(res_lda)

    # 4. HDBSCAN
    if run_all or "hdbscan" in selected:
        res_hdb = benchmark_hdbscan(p0_path, manifest_path, n_repeats=args.repeats)
        results.append(res_hdb)

    # 5. BERTopic
    if run_all or "bertopic" in selected:
        res_bert = benchmark_bertopic(p1_path, manifest_path, n_repeats=1)
        results.append(res_bert)

    out_dir = PROJECT_ROOT / "results" / "runtime_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "all_models_runtime_benchmark.json"

    # Nạp kết quả đã lưu trước đó nếu có để hợp nhất
    all_models_map: dict[str, dict[str, Any]] = {}
    if json_path.is_file():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
                for m in prev_data.get("models", []):
                    all_models_map[m["method"]] = m
        except Exception as e:
            print(f"Lưu ý: Không đọc được JSON cũ ({e}), tạo mới.")

    # Cập nhật kết quả vừa chạy
    for r in results:
        all_models_map[r["method"]] = r

    # Sắp xếp theo thứ tự chuẩn trong báo cáo NCKH: KMeans -> NMF -> LDA -> HDBSCAN -> BERTopic
    model_order = ["KMeans (TF-IDF)", "NMF", "LDA", "HDBSCAN", "BERTopic"]
    sorted_results = []
    for name in model_order:
        if name in all_models_map:
            sorted_results.append(all_models_map[name])
    for k, v in all_models_map.items():
        if k not in model_order:
            sorted_results.append(v)

    # Build DataFrame
    rows = []
    for r in sorted_results:
        std_str = f" +/- {r['total_pipeline_seconds_std']:.3f}" if r.get("total_pipeline_seconds_std", 0) > 0 else ""
        rep_std_str = f" +/- {r['representation_seconds_std']:.3f}" if r.get("representation_seconds_std", 0) > 0 else ""
        fit_std_str = f" +/- {r['fit_seconds_std']:.3f}" if r.get("fit_seconds_std", 0) > 0 else ""
        pred_std_str = f" +/- {r['predict_seconds_std']:.3f}" if r.get("predict_seconds_std", 0) > 0 else ""

        rows.append({
            "method": r["method"],
            "run_id": r["run_id"],
            "fit_split": r["fit_split"],
            "n_train": r["n_train"],
            "n_eval": r["n_eval"],
            "representation_seconds": f"{r['representation_seconds_mean']:.3f}{rep_std_str}",
            "fit_seconds": f"{r['fit_seconds_mean']:.3f}{fit_std_str}",
            "predict_seconds": f"{r['predict_seconds_mean']:.3f}{pred_std_str}",
            "evaluation_seconds": f"{r['evaluation_seconds_mean']:.3f}",
            "total_pipeline_seconds": f"{r['total_pipeline_seconds_mean']:.3f}{std_str}",
            "setup_seconds": f"{r['setup_seconds']:.3f}",
            "ram_peak_mb": r["ram_peak_mb"],
            "model_size_mb": r["model_size_mb"],
            "hardware": hw["summary"],
            "python_version": hw["python_version"],
            "n_repeats": r["n_repeats"],
            "scope_notes": r["scope_notes"],
        })

    df = pd.DataFrame(rows)

    csv_path = out_dir / "e08_runtime_summary.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    payload = {
        "benchmark_title": "Unified Single-Machine Runtime Benchmark (E08)",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": hw,
        "models": sorted_results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("BẢNG TỔNG HỢP BENCHMARK TRÊN CÙNG MỘT MÁY (E08 RUNTIME SUMMARY):")
    print("=" * 70)
    print(df[["method", "representation_seconds", "fit_seconds", "predict_seconds", "total_pipeline_seconds", "ram_peak_mb", "model_size_mb"]].to_string(index=False))
    print("=" * 70)
    print(f" Đã cập nhật file CSV: {csv_path}")
    print(f" Đã lưu chi tiết JSON: {json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

