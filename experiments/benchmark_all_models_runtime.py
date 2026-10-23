"""E08: benchmark năm pipeline trên một máy, Train -> Test, không sửa artifact gốc."""

from __future__ import annotations

import argparse
import json
import os
import platform
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.data.dataset_loader import load_split_dataset

ROOT = Path(__file__).resolve().parents[1]
# Benchmark chỉ dùng model đã cache; tránh retry mạng làm sai setup time.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
MODELS = ("kmeans", "nmf", "lda", "hdbscan", "bertopic")
SPLITS = ROOT / "data/splits/split_manifest.csv"


def config(name: str) -> dict:
    path = ROOT / "artifacts/configs" / f"{name}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def dataset(variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = ROOT / "data/processed" / f"{variant}.csv"
    train = load_split_dataset(path, SPLITS, "train")
    test = load_split_dataset(path, SPLITS, "test")
    if len(train) != 6493 or len(test) != 1390:
        raise ValueError(f"Split {variant} không đúng 6493/1390 bài")
    if set(train.article_id) & set(test.article_id):
        raise ValueError("Train và Test trùng article_id")
    if test.category_gold.isna().any():
        raise ValueError("Test thiếu category_gold")
    return train, test


def measure(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, time.perf_counter() - start


def score(labels, gold):
    labels = np.asarray(labels, dtype=int)
    if len(labels) != len(gold):
        raise ValueError("Số nhãn dự đoán không khớp Test")
    return {
        "ari": float(adjusted_rand_score(gold, labels)),
        "nmi": float(normalized_mutual_info_score(gold, labels)),
        "outlier_rate": float(np.mean(labels == -1)) if np.any(labels == -1) else None,
    }


def lexical(method: str, train: pd.DataFrame, test: pd.DataFrame):
    if method == "kmeans":
        cfg = config("kmeans")
        # M01 vectorizer đã chốt; cấu hình KMeans riêng chỉ định K=9.
        rep = config("tfidf")["vectorizer"]
        vectorizer = TfidfVectorizer(
            max_features=rep.get("max_features"), min_df=rep.get("min_df", 1),
            max_df=rep.get("max_df", 1.0), ngram_range=tuple(rep.get("ngram_range", [1, 1])),
            sublinear_tf=rep.get("sublinear_tf", False), norm=rep.get("norm", "l2"),
            token_pattern=rep["token_pattern"],
        )
        model = KMeans(n_clusters=9, init="k-means++", n_init=10, max_iter=300, random_state=42)
    elif method == "nmf":
        cfg = config("nmf")
        rep, params = cfg["representation"], cfg["model"]
        vectorizer = TfidfVectorizer(
            ngram_range=tuple(rep["ngram_range"]), min_df=rep["min_df"],
            max_df=rep["max_df"], max_features=rep["max_features"],
            sublinear_tf=rep["sublinear_tf"], token_pattern=rep["token_pattern"],
        )
        model = NMF(n_components=10, init=params["init"], solver=params["solver"],
                    beta_loss=params["beta_loss"], max_iter=params["max_iter"],
                    tol=params["tolerance"], random_state=params["random_seed"])
    else:
        cfg = config("lda")
        rep, params = cfg["vectorizer"], cfg["lda"]
        vectorizer = CountVectorizer(
            max_features=rep["max_features"], min_df=rep["min_df"],
            max_df=rep["max_df"], ngram_range=tuple(rep["ngram_range"]),
            token_pattern=rep["token_pattern"],
        )
        model = LatentDirichletAllocation(
            n_components=15, learning_method=params["learning_method"],
            max_iter=params["max_iter"], random_state=42, n_jobs=params["n_jobs"],
        )
    train_text = train["text"].fillna("").astype(str).tolist()
    test_text = test["text"].fillna("").astype(str).tolist()
    x_train, train_rep = measure(lambda: vectorizer.fit_transform(train_text))
    x_test, test_rep = measure(lambda: vectorizer.transform(test_text))
    _, fit = measure(lambda: model.fit(x_train))
    if method == "kmeans":
        labels, predict = measure(lambda: model.predict(x_test))
    else:
        weights, predict = measure(lambda: model.transform(x_test))
        labels = weights.argmax(axis=1)
    metrics, evaluation = measure(lambda: score(labels, test.category_gold))
    return {"representation_seconds": train_rep + test_rep, "fit_seconds": fit,
            "predict_seconds": predict, "evaluation_seconds": evaluation,
            "test_metrics": metrics, "representation_device": "cpu"}


def semantic(method: str, train: pd.DataFrame, test: pd.DataFrame, device: str):
    from sentence_transformers import SentenceTransformer
    if method == "hdbscan":
        from hdbscan import HDBSCAN, approximate_predict
        from umap import UMAP
        cfg = config("hdbscan")
        texts = lambda frame: [
            f"{title}. {' '.join(str(content).split()[:60])}" for title, content in zip(
                frame.processed_title.fillna(""), frame.processed_content.fillna(""))
        ]
        name = config("embeddings")["runs"]["minilm_title_content"]["model_name"]
    else:
        from src.models.bertopic_pipeline import build_bertopic_model
        cfg = config("bertopic")
        run = next(x for x in config("bertopic_experiments")["runs"] if x["run_id"] == "e1_vietnamese_sbert")
        cfg["embedding"].update(run["embedding"])
        cfg["umap"].update(run["umap"])
        cfg["hdbscan"].update(run["hdbscan"])
        cfg["vectorizer"]["stopwords_file"] = str(ROOT / "artifacts/configs/vietnamese_stopwords.txt")
        texts = lambda frame: frame["text"].fillna("").astype(str).tolist()
        name = run["embedding"]["model_name"]
    encoder, setup = measure(lambda: SentenceTransformer(name, device=device))
    train_text, test_text = texts(train), texts(test)
    batch = 16 if method == "hdbscan" else 32
    encode = lambda values: encoder.encode(values, batch_size=batch, show_progress_bar=False,
                                             normalize_embeddings=True)
    train_vectors, train_rep = measure(lambda: encode(train_text))
    test_vectors, test_rep = measure(lambda: encode(test_text))
    if method == "hdbscan":
        reducer = UMAP(**cfg["umap"])
        cluster = HDBSCAN(min_cluster_size=15, min_samples=5, metric="euclidean",
                          cluster_selection_method="eom", prediction_data=True)
        def fit_model():
            reduced = reducer.fit_transform(train_vectors)
            cluster.fit(reduced)
        _, fit = measure(fit_model)
        labels, predict = measure(lambda: approximate_predict(cluster, reducer.transform(test_vectors))[0])
    else:
        topic = build_bertopic_model(cfg, embedding_model=encoder)
        _, fit = measure(lambda: topic.fit_transform(train_text, embeddings=train_vectors))
        prediction, predict = measure(lambda: topic.transform(test_text, embeddings=test_vectors))
        labels = prediction[0]
    metrics, evaluation = measure(lambda: score(labels, test.category_gold))
    return {"setup_seconds": setup, "representation_seconds": train_rep + test_rep,
            "fit_seconds": fit, "predict_seconds": predict,
            "evaluation_seconds": evaluation, "test_metrics": metrics,
            "representation_device": device, "batch_size": batch}


def benchmark(method: str, device: str):
    variant = {"kmeans": "p2_segmented", "nmf": "p3_no_stopwords",
               "lda": "p2_segmented", "hdbscan": "p0_minimal",
               "bertopic": "p1_normalized"}[method]
    train, test = dataset(variant)
    process = psutil.Process()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rss = [process.memory_info().rss]
    stop = threading.Event()
    def sample():
        while not stop.wait(0.05):
            rss.append(process.memory_info().rss)
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    try:
        result = (lexical(method, train, test) if method in ("kmeans", "nmf", "lda")
                  else semantic(method, train, test, device))
    finally:
        stop.set()
        monitor.join()
    result.update(method=method, preprocessing_variant=variant, train_count=len(train),
                  test_count=len(test), peak_process_rss_mb=max(rss) / 1048576,
                  peak_gpu_allocated_mb=(torch.cuda.max_memory_allocated() / 1048576
                                         if device == "cuda" else None),
                  runtime_seconds=sum(result[k] for k in (
                      "representation_seconds", "fit_seconds", "predict_seconds", "evaluation_seconds")))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all", help="all hoặc danh sách cách nhau bằng dấu phẩy")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--check", action="store_true", help="Chỉ kiểm tra môi trường và dữ liệu")
    args = parser.parse_args()
    selected = list(MODELS) if args.models == "all" else args.models.split(",")
    if not selected or any(name not in MODELS for name in selected) or len(set(selected)) != len(selected):
        parser.error("--models phải gồm tên hợp lệ, không trùng")
    if args.repeats < 1:
        parser.error("--repeats phải >= 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA chưa sẵn sàng trong môi trường Python này")
    for method in selected:
        variant = {"kmeans": "p2_segmented", "nmf": "p3_no_stopwords",
                   "lda": "p2_segmented", "hdbscan": "p0_minimal",
                   "bertopic": "p1_normalized"}[method]
        train, test = dataset(variant)
        print(f"{method}: {variant}, Train={len(train)}, Test={len(test)}", flush=True)
    if args.check:
        print(f"GPU: {torch.cuda.get_device_name(0) if args.device == 'cuda' else 'CPU'}")
        return
    output = ROOT / "results/runtime_benchmark" / datetime.now().strftime("e08_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    environment = {"platform": platform.platform(), "cpu": platform.processor(),
                   "ram_gb": psutil.virtual_memory().total / 1073741824,
                   "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                   "python": platform.python_version(), "torch": torch.__version__,
                   "cuda": torch.version.cuda, "device": args.device}
    rows = []
    for method in selected:
        for repeat in range(1, args.repeats + 1):
            print(f"E08 {method} {repeat}/{args.repeats}", flush=True)
            row = benchmark(method, args.device)
            row["repeat"] = repeat
            rows.append(row)
            (output / "runs.json").write_text(json.dumps({"environment": environment, "runs": rows},
                                                  ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "runs.csv", index=False, encoding="utf-8-sig")
    summary_fields = ["runtime_seconds", "representation_seconds", "fit_seconds",
                      "predict_seconds", "evaluation_seconds", "peak_process_rss_mb",
                      "peak_gpu_allocated_mb"]
    summary = frame.groupby("method", as_index=False).agg(
        **{f"{field}_mean": (field, "mean") for field in summary_fields},
        runtime_std_seconds=("runtime_seconds", "std"),
        n_repeats=("repeat", "count"))
    summary["runtime_std_seconds"] = summary["runtime_std_seconds"].fillna(0.0)
    summary.to_csv(output / "e08_runtime_summary.csv", index=False, encoding="utf-8-sig")
    print(f"E08 output: {output}")


if __name__ == "__main__":
    main()
