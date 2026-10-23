"""Suy luận HDBSCAN E07 trên dữ liệu tháng 10 đã khóa."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.features.embeddings import create_embeddings
from src.models.hdbscan import predict_hdbscan_validation


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/ood/oct2026"
RESULT_DIR = ROOT / "results/hdbscan/ood"

EMBEDDING_RUN = "minilm_title_content"
EXPERIMENT = "eom_15_5"

UMAP_PATH = (
    ROOT
    / "artifacts/hdbscan/validation"
    / f"{EMBEDDING_RUN}_umap.joblib"
)
HDBSCAN_PATH = (
    ROOT
    / "artifacts/hdbscan/validation"
    / f"{EMBEDDING_RUN}_{EXPERIMENT}_hdbscan.joblib"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_articles() -> tuple[pd.DataFrame, dict]:
    manifest_path = DATA_DIR / "manifest.json"
    data_path = DATA_DIR / "news_articles_p0.csv"

    if not manifest_path.is_file() or not data_path.is_file():
        raise FileNotFoundError(
            "Thiếu manifest hoặc news_articles_p0.csv của E07."
        )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    expected_hash = manifest["files"]["news_articles_p0.csv"]
    if sha256_file(data_path) != expected_hash:
        raise ValueError(
            "Hash của news_articles_p0.csv khác manifest."
        )

    articles = pd.read_csv(
        data_path,
        encoding="utf-8-sig",
        dtype={"article_id": str},
    )

    required = {
        "article_id",
        "title",
        "published_at",
        "source",
        "url",
        "category_gold",
        "processed_title",
        "processed_content",
    }
    missing = required - set(articles.columns)
    if missing:
        raise ValueError(f"Dữ liệu E07 thiếu cột: {sorted(missing)}")

    if len(articles) != manifest["n_articles"]:
        raise ValueError("Số bài không khớp manifest.")

    if articles.empty or articles["article_id"].duplicated().any():
        raise ValueError("Dữ liệu rỗng hoặc article_id trùng.")

    if articles[
        ["processed_title", "processed_content", "category_gold"]
    ].isna().any().any():
        raise ValueError("Dữ liệu E07 có văn bản P0 hoặc nhãn rỗng.")

    return articles, manifest


def group_metrics(
    assignments: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    rows = []

    for value, group in assignments.groupby(
        group_column, dropna=False
    ):
        n_articles = len(group)
        n_noise = int(group["cluster_id"].eq(-1).sum())

        rows.append(
            {
                group_column: value,
                "n_articles": n_articles,
                "n_noise": n_noise,
                "noise_percent": round(
                    100 * n_noise / n_articles, 2
                ),
                "coverage_percent": round(
                    100 * (n_articles - n_noise) / n_articles,
                    2,
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("n_articles", ascending=False)
        .reset_index(drop=True)
    )


def select_review_candidates(
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Chọn bài để đọc thủ công; chưa phải kết luận định tính."""
    assigned = assignments.loc[
        assignments["cluster_id"].ne(-1)
    ].copy()
    noise = assignments.loc[
        assignments["cluster_id"].eq(-1)
    ].copy()

    largest_clusters = (
        assigned["cluster_id"]
        .value_counts()
        .head(5)
        .index
    )

    representative = (
        assigned.loc[
            assigned["cluster_id"].isin(largest_clusters)
        ]
        .sort_values(
            "membership_strength",
            ascending=False,
        )
        .groupby("cluster_id", group_keys=False)
        .head(2)
        .copy()
    )
    representative["selection_type"] = (
        "high_membership_in_large_cluster"
    )

    if len(noise) > 10:
        noise = noise.sample(n=10, random_state=42)
    noise["selection_type"] = "noise_review_candidate"

    return pd.concat(
        [representative, noise],
        ignore_index=True,
    )


def main() -> None:
    if RESULT_DIR.exists() and any(RESULT_DIR.iterdir()):
        raise FileExistsError(
            f"{RESULT_DIR} đã có kết quả. "
            "Kiểm tra trước khi chạy lại để tránh ghi đè."
        )

    for path in (UMAP_PATH, HDBSCAN_PATH):
        if not path.is_file():
            raise FileNotFoundError(
                f"Thiếu model đã lưu: {path}"
            )

    articles, dataset_manifest = load_frozen_articles()

    embedding_config_path = ROOT / "artifacts/configs/embeddings.yaml"
    config = yaml.safe_load(
        embedding_config_path.read_text(encoding="utf-8")
    )
    run_config = config["runs"][EMBEDDING_RUN]

    if (
        config["dataset"]["title_col"] != "processed_title"
        or config["dataset"]["content_col"] != "processed_content"
        or run_config["content_words"] != 60
    ):
        raise ValueError(
            "Cấu hình MiniLM không còn là title + 60 từ P0."
        )

    print(f"Đang mã hóa {len(articles)} bài E07...")
    encoder = SentenceTransformer(run_config["model_name"])

    embeddings = create_embeddings(
        articles,
        encoder,
        model_name=run_config["model_name"],
        data_version=dataset_manifest["dataset"],
        content_words=run_config["content_words"],
        title_col=config["dataset"]["title_col"],
        content_col=config["dataset"]["content_col"],
        batch_size=config["encoding"]["batch_size"],
    )

    dimensions = len(embeddings["embedding"].iloc[0])
    if dimensions != run_config["expected_dimensions"]:
        raise ValueError(
            f"Embedding có {dimensions} chiều; "
            f"cấu hình yêu cầu {run_config['expected_dimensions']}."
        )

    # Chỉ nạp hai model đã fit trên train; không fit trên E07.
    reducer = joblib.load(UMAP_PATH)
    clusterer = joblib.load(HDBSCAN_PATH)

    if getattr(reducer, "n_components", None) != 5:
        raise ValueError("UMAP đã lưu không phải model 5 chiều.")
    if not getattr(clusterer, "prediction_data", False):
        raise ValueError(
            "HDBSCAN đã lưu thiếu prediction_data."
        )
    if (
        getattr(clusterer, "cluster_selection_method", None)
        != "eom"
    ):
        raise ValueError(
            "HDBSCAN đã lưu không dùng cách chọn cụm eom."
        )

    predicted = predict_hdbscan_validation(
        embeddings,
        reducer,
        clusterer,
    )

    if (
        predicted["article_id"].tolist()
        != articles["article_id"].tolist()
    ):
        raise ValueError(
            "Thứ tự article_id thay đổi sau khi dự đoán."
        )

    assignments = articles[
        [
            "article_id",
            "title",
            "published_at",
            "source",
            "url",
            "category_gold",
        ]
    ].copy()

    assignments.insert(1, "split", "ood_oct2026")
    assignments["embedding_run"] = EMBEDDING_RUN
    assignments["experiment"] = EXPERIMENT
    assignments["cluster_id"] = (
        predicted["cluster_id"].to_numpy()
    )
    assignments["membership_strength"] = (
        predicted["membership_strength"].to_numpy()
    )

    n_articles = len(assignments)
    n_noise = int(
        assignments["cluster_id"].eq(-1).sum()
    )

    # Tính trên toàn bộ E07; -1 được xem là một nhãn dự đoán.
    ari = adjusted_rand_score(
        assignments["category_gold"],
        assignments["cluster_id"],
    )
    nmi = normalized_mutual_info_score(
        assignments["category_gold"],
        assignments["cluster_id"],
    )

    metrics = pd.DataFrame(
        [
            {
                "embedding_run": EMBEDDING_RUN,
                "experiment": EXPERIMENT,
                "split": "ood_oct2026",
                "n_articles": n_articles,
                "n_noise": n_noise,
                "noise_percent": round(
                    100 * n_noise / n_articles, 2
                ),
                "coverage_percent": round(
                    100 * (n_articles - n_noise) / n_articles,
                    2,
                ),
                "n_clusters_assigned": int(
                    assignments.loc[
                        assignments["cluster_id"].ne(-1),
                        "cluster_id",
                    ].nunique()
                ),
                "ari": round(float(ari), 4),
                "nmi": round(float(nmi), 4),
            }
        ]
    )

    by_source = group_metrics(assignments, "source")
    by_category = group_metrics(
        assignments, "category_gold"
    )
    review_candidates = select_review_candidates(
        assignments
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "ood_assignments.csv": assignments,
        "ood_metrics.csv": metrics,
        "ood_by_source.csv": by_source,
        "ood_by_category.csv": by_category,
        "ood_review_candidates.csv": review_candidates,
    }

    for filename, frame in outputs.items():
        frame.to_csv(
            RESULT_DIR / filename,
            index=False,
            encoding="utf-8-sig",
        )

    run_manifest = {
        "dataset_manifest": (
            "data/ood/oct2026/manifest.json"
        ),
        "dataset_p0_sha256": (
            dataset_manifest["files"][
                "news_articles_p0.csv"
            ]
        ),
        "embedding_run": EMBEDDING_RUN,
        "experiment": EXPERIMENT,
        "embedding_config": (
            "artifacts/configs/embeddings.yaml"
        ),
        "umap_model": str(
            UMAP_PATH.relative_to(ROOT)
        ).replace("\\", "/"),
        "umap_sha256": sha256_file(UMAP_PATH),
        "hdbscan_model": str(
            HDBSCAN_PATH.relative_to(ROOT)
        ).replace("\\", "/"),
        "hdbscan_sha256": sha256_file(HDBSCAN_PATH),
        "prediction_method": (
            "predict_hdbscan_validation"
        ),
        "fit_on_ood": False,
        "ari_nmi_include_noise_label": True,
        "output_sha256": {
            filename: sha256_file(RESULT_DIR / filename)
            for filename in outputs
        },
    }

    (RESULT_DIR / "run_manifest.json").write_text(
        json.dumps(
            run_manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(metrics.to_string(index=False))
    print(f"Đã lưu kết quả tại: {RESULT_DIR}")


if __name__ == "__main__":
    main()