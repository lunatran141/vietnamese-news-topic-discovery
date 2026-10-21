"""Evaluate the locked MiniLM EOM HDBSCAN run on the held-out test split."""

import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from experiments.hdbscan.run_m11_hdbscan import get_config_hash
from src.data.dataset_loader import load_split_dataset
from src.features.embeddings import create_embeddings
from src.models.hdbscan import predict_hdbscan_validation


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "minilm_title_content_eom_15_5"
EMBEDDING_RUN = "minilm_title_content"
EXPERIMENT = "eom_15_5"
OUTPUT_DIR = ROOT / "results/hdbscan/test"
ASSIGNMENTS_PATH = OUTPUT_DIR / f"{RUN_ID}_test_assignments.csv"
METRICS_PATH = OUTPUT_DIR / f"{RUN_ID}_test_metrics.json"


def main() -> None:
    with (ROOT / "configs/embeddings.yaml").open(encoding="utf-8") as file:
        m10 = yaml.safe_load(file)
    with (ROOT / "configs/hdbscan.yaml").open(encoding="utf-8") as file:
        m11 = yaml.safe_load(file)
    config_hash = get_config_hash(m11, m10)

    metadata = json.loads(
        (ROOT / f"artifacts/common/hdbscan/{RUN_ID}/run_metadata.json")
        .read_text(encoding="utf-8")
    )
    if (metadata["run_id"], metadata["config_hash"], metadata["fit_split"]) != (
        RUN_ID, config_hash, "train"
    ):
        raise ValueError("Run P01 or its configuration is not locked as expected")

    dataset = m11["dataset"]
    dataset_path = ROOT / dataset["path"]
    split_path = ROOT / dataset["split_manifest_path"]
    train = load_split_dataset(dataset_path, split_path, "train").reset_index(drop=True)
    val = load_split_dataset(dataset_path, split_path, "val").reset_index(drop=True)
    test = load_split_dataset(dataset_path, split_path, "test").reset_index(drop=True)
    if (len(train), len(val), len(test)) != (6493, 1391, 1390):
        raise ValueError("Unexpected train/val/test split sizes")
    if len(set(test["article_id"])) != len(test):
        raise ValueError("Duplicate test article_id")
    if set(test["article_id"]) & (set(train["article_id"]) | set(val["article_id"])):
        raise ValueError("Test article_id overlaps train or validation")
    if test["category_gold"].isna().any():
        raise ValueError("Missing test category_gold")

    model_dir = ROOT / "artifacts/hdbscan/validation"
    load_start = perf_counter()
    reducer = joblib.load(model_dir / f"{EMBEDDING_RUN}_umap.joblib")
    clusterer = joblib.load(model_dir / f"{RUN_ID}_hdbscan.joblib")
    encoder_config = m10["runs"][EMBEDDING_RUN]
    encoder = SentenceTransformer(encoder_config["model_name"])
    model_load_seconds = perf_counter() - load_start

    original = pd.read_csv(
        ROOT / "results/hdbscan/minilm_title_content_assignments.csv",
        dtype={"article_id": str},
    )
    original = original.loc[
        original["embedding_run"].eq(EMBEDDING_RUN)
        & original["experiment"].eq(EXPERIMENT)
    ]
    if len(original) != len(train) or not original["config_hash"].eq(config_hash).all():
        raise ValueError("Saved train assignments do not match the locked run")
    if set(original["article_id"]) != set(train["article_id"]):
        raise ValueError("Saved train article IDs do not match")
    train_labels = original.set_index("article_id").loc[
        train["article_id"], "cluster_id"
    ].to_numpy(dtype=int)
    if not np.array_equal(np.asarray(clusterer.labels_, dtype=int), train_labels):
        raise ValueError("Saved HDBSCAN model differs from M11 train assignments")

    representation_start = perf_counter()
    embeddings = create_embeddings(
        test,
        encoder,
        model_name=encoder_config["model_name"],
        data_version=m10["dataset"]["data_version"],
        content_words=encoder_config["content_words"],
        title_col=m10["dataset"]["title_col"],
        content_col=m10["dataset"]["content_col"],
        batch_size=m10["encoding"]["batch_size"],
    )
    test_representation_seconds = perf_counter() - representation_start
    if embeddings["article_id"].tolist() != test["article_id"].tolist():
        raise ValueError("Test embedding order changed")

    predict_start = perf_counter()
    predictions = predict_hdbscan_validation(embeddings, reducer, clusterer)
    test_predict_seconds = perf_counter() - predict_start
    if predictions["article_id"].tolist() != test["article_id"].tolist():
        raise ValueError("Test prediction order changed")
    if not set(predictions["cluster_id"]).issubset(set(train_labels)):
        raise ValueError("Test has a cluster ID that was not fitted on train")

    evaluation_start = perf_counter()
    paired = predictions.merge(
        test[["article_id", "category_gold"]],
        on="article_id", how="inner", validate="one_to_one",
    )
    if len(paired) != len(test):
        raise ValueError("Cannot match every test prediction to category_gold")
    n_outliers = int(paired["cluster_id"].eq(-1).sum())
    outlier_rate = n_outliers / len(paired)
    ari = adjusted_rand_score(paired["category_gold"], paired["cluster_id"])
    nmi = normalized_mutual_info_score(paired["category_gold"], paired["cluster_id"])
    evaluation_seconds = perf_counter() - evaluation_start

    predictions.insert(1, "split", "test")
    predictions["embedding_run"] = EMBEDDING_RUN
    predictions["experiment"] = EXPERIMENT
    predictions["config_hash"] = config_hash
    metrics = {
        "run_id": RUN_ID,
        "config_hash": config_hash,
        "fit_split": "train",
        "split": "test",
        "test_count": len(test),
        "test_outlier_count": n_outliers,
        "test_outlier_rate": outlier_rate,
        "test_coverage": 1 - outlier_rate,
        "test_ari": ari,
        "test_nmi": nmi,
        "metric_scope": "All test articles, including outlier label -1; category_gold used only after prediction",
        "model_load_seconds": model_load_seconds,
        "test_representation_seconds": test_representation_seconds,
        "test_predict_seconds": test_predict_seconds,
        "evaluation_seconds": evaluation_seconds,
        "runtime_seconds": test_representation_seconds + test_predict_seconds,
        "runtime_scope": "MiniLM encode test + UMAP transform + HDBSCAN approximate_predict test; excludes model load, evaluation and I/O",
        "assignments_path": ASSIGNMENTS_PATH.relative_to(ROOT).as_posix(),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(ASSIGNMENTS_PATH, index=False, encoding="utf-8-sig")
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {ASSIGNMENTS_PATH.relative_to(ROOT)} ({len(predictions)} articles)")
    print(f"Saved {METRICS_PATH.relative_to(ROOT)} (ARI={ari:.4f}, NMI={nmi:.4f})")


if __name__ == "__main__":
    main()
