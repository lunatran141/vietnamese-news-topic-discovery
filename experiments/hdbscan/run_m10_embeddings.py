"""Tạo và lưu hai bộ sentence embeddings cho task M10."""

from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from src.data.dataset_loader import load_split_dataset
from src.features.embeddings import (
    create_embeddings,
    load_embedding_artifact,
    save_embedding_artifact,
    build_embedding_metadata,
)

CONFIG_PATH = Path("configs/embeddings.yaml")
RUN_NAMES = (
    "minilm_title_content",
    "mpnet_title_content",
    "minilm_title_only",
    "mpnet_title_only",
)

def main() -> None:
    """Chạy hai mô hình trên cùng các bài và kiểm tra artifact đã lưu."""
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    dataset_config = config["dataset"]
    encoding_config = config["encoding"]

    if encoding_config["normalize_embeddings"] is not True:
        raise ValueError(
            "create_embeddings hiện luôn chuẩn hóa vector; "
            "normalize_embeddings trong YAML phải là true"
        )

    train = load_split_dataset(
        dataset_config["path"],
        dataset_config["split_manifest_path"],
        dataset_config["split"]
    )

    sample_size = dataset_config["sample_size"]

    if sample_size is None:
        articles = train.reset_index(drop=True)
    else:
        if not isinstance(sample_size, int) or not 0 < sample_size <= len(train):
            raise ValueError("sample_size phải là null hoặc số từ 1 đến số bài train")

        articles = train.sample(
            n=sample_size,
            random_state=dataset_config["sample_random_state"]).reset_index(drop=True)

    article_ids = articles["article_id"].tolist()

    print("Số bài trong tập train:", len(train))
    print("Số bài được mã hóa:", len(articles))
    print("article_id đầu tiên:", article_ids[0])

    for run_name in RUN_NAMES:
        run_config = config["runs"][run_name]
        model_name = run_config["model_name"]
        artifact_path = Path(run_config["artifact_path"])

        print(f"\nĐang chạy: {run_name}")
        print("Mô hình:", model_name)

        model = SentenceTransformer(model_name)

        embedding_table = create_embeddings(
            articles,
            model,
            model_name=model_name,
            data_version=dataset_config["data_version"],
            content_words=run_config["content_words"],
            title_col=dataset_config["title_col"],
            content_col=dataset_config["content_col"],
            batch_size=encoding_config["batch_size"],
        )

        save_embedding_artifact(embedding_table, artifact_path, metadata=build_embedding_metadata(config, run_name),)
        loaded = load_embedding_artifact(artifact_path)

        vectors = np.stack(loaded["embedding"].to_numpy())
        expected_shape = (
            len(articles),
            run_config["expected_dimensions"],
        )

        if loaded["article_id"].tolist() != article_ids:
            raise AssertionError(
                f"{run_name}: article_id hoặc thứ tự bài bị thay đổi"
            )

        if vectors.shape != expected_shape:
            raise AssertionError(
                f"{run_name}: kích thước {vectors.shape}, "
                f"mong đợi {expected_shape}"
            )

        if not np.isfinite(vectors).all():
            raise AssertionError(f"{run_name}: vector chứa NaN hoặc Infinity")

        if loaded["model_name"].nunique() != 1:
            raise AssertionError(f"{run_name}: artifact có nhiều model_name")

        if loaded["model_name"].iloc[0] != model_name:
            raise AssertionError(f"{run_name}: model_name không đúng")

        if loaded["data_version"].iloc[0] != dataset_config["data_version"]:
            raise AssertionError(f"{run_name}: data_version không đúng")

        print("Kích thước vector:", vectors.shape)
        print("Đã lưu và tải lại:", artifact_path)

    print("\nM10 embedding hoàn tất cho cả hai mô hình.")

if __name__ == "__main__":
    main()
