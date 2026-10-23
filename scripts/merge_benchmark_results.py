import json
from pathlib import Path
import sys
import pandas as pd

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
out_dir = PROJECT_ROOT / "results" / "runtime_benchmark"
json_path = out_dir / "all_models_runtime_benchmark.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 5 mô hình đo thực tế trên cùng một máy
km = {
    "method": "KMeans (TF-IDF)",
    "run_id": "kmeans_baseline",
    "fit_split": "train",
    "n_train": 6493,
    "n_eval": 1390,
    "setup_seconds": 0.0,
    "representation_seconds_mean": 17.826,
    "representation_seconds_std": 0.373,
    "fit_seconds_mean": 3.920,
    "fit_seconds_std": 0.061,
    "predict_seconds_mean": 0.0078,
    "predict_seconds_std": 0.0004,
    "evaluation_seconds_mean": 0.0038,
    "evaluation_seconds_std": 0.0021,
    "total_pipeline_seconds_mean": 21.757,
    "total_pipeline_seconds_std": 0.431,
    "ram_peak_mb": 58.9,
    "model_size_mb": 2.25,
    "n_repeats": 3,
    "scope_notes": "TF-IDF (20k terms) fit Train + transform Test + KMeans (K=9) fit + predict Test + eval; excludes disk I/O & plotting",
}

nmf = {
    "method": "NMF",
    "run_id": "nmf_p3_k10_seed42_dashboard_v2",
    "fit_split": "train+val",
    "n_train": 7884,
    "n_eval": 1390,
    "setup_seconds": 0.0,
    "representation_seconds_mean": 34.304,
    "representation_seconds_std": 0.568,
    "fit_seconds_mean": 5.022,
    "fit_seconds_std": 0.175,
    "predict_seconds_mean": 0.0549,
    "predict_seconds_std": 0.0118,
    "evaluation_seconds_mean": 0.1465,
    "evaluation_seconds_std": 0.0068,
    "total_pipeline_seconds_mean": 39.527,
    "total_pipeline_seconds_std": 0.451,
    "ram_peak_mb": 363.1,
    "model_size_mb": 22.26,
    "n_repeats": 3,
    "scope_notes": "TF-IDF (50k terms) fit Train+Val + transform Test + NMF (K=10) fit + predict Test + eval; excludes disk I/O & plotting",
}

lda = {
    "method": "LDA",
    "run_id": "lda_k15_seed42",
    "fit_split": "train+val",
    "n_train": 7884,
    "n_eval": 1390,
    "setup_seconds": 0.0,
    "representation_seconds_mean": 8.964,
    "representation_seconds_std": 1.408,
    "fit_seconds_mean": 16.739,
    "fit_seconds_std": 2.973,
    "predict_seconds_mean": 0.2008,
    "predict_seconds_std": 0.0078,
    "evaluation_seconds_mean": 0.0452,
    "evaluation_seconds_std": 0.0048,
    "total_pipeline_seconds_mean": 25.949,
    "total_pipeline_seconds_std": 1.577,
    "ram_peak_mb": 77.0,
    "model_size_mb": 3.87,
    "n_repeats": 3,
    "scope_notes": "BoW (15k terms) fit Train+Val + transform Test + LDA (K=15) fit + predict Test + eval; excludes disk I/O & plotting",
}

hdb = {
    "method": "HDBSCAN",
    "run_id": "minilm_title_content_eom_15_5",
    "fit_split": "train",
    "n_train": 6493,
    "n_eval": 1390,
    "setup_seconds": 3.848,
    "representation_seconds_mean": 139.245,
    "representation_seconds_std": 0.0,
    "fit_seconds_mean": 68.260,
    "fit_seconds_std": 0.0,
    "predict_seconds_mean": 37.021,
    "predict_seconds_std": 0.0,
    "evaluation_seconds_mean": 0.016,
    "evaluation_seconds_std": 0.0,
    "total_pipeline_seconds_mean": 244.542,
    "total_pipeline_seconds_std": 0.0,
    "ram_peak_mb": 1059.2,
    "model_size_mb": 36.39,
    "n_repeats": 1,
    "scope_notes": "MiniLM encode + UMAP 5D fit Train + HDBSCAN fit Train + UMAP transform Test + approximate_predict Test + eval; excludes model download & disk I/O",
}

bert = [m for m in data["models"] if m["method"] == "BERTopic"][0]

all_models = [km, nmf, lda, hdb, bert]
data["models"] = all_models

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

hw = data["hardware"]
rows = []
for r in all_models:
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
print("=== HOÀN TẤT TỔNG HỢP 5 MÔ HÌNH ===")
print(df[["method", "representation_seconds", "fit_seconds", "predict_seconds", "total_pipeline_seconds", "setup_seconds", "ram_peak_mb", "model_size_mb"]].to_string(index=False))
