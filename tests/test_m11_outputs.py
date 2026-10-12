"""Kiểm tra tính đầy đủ và nhất quán của các kết quả M11 đã lưu."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.hdbscan.run_m11_hdbscan import get_config_hash
from src.data.dataset_loader import load_split_dataset


ROOT = Path(__file__).resolve().parents[1]

EMBEDDING_NAMES = (
    "minilm_title_content",
    "mpnet_title_content",
    "minilm_title_only",
    "mpnet_title_only",
)

@pytest.fixture(scope="module")
def context():
    with (ROOT / "configs/hdbscan.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    dataset = config["dataset"]
    train = load_split_dataset(
        ROOT / dataset["path"],
        ROOT / dataset["split_manifest_path"],
        dataset["split"],
    )
    assert len(train) == dataset["expected_articles"]

    output_dir = ROOT / config["output_dir"]
    summary = pd.read_csv(
        output_dir / "m11_summary.csv",
        encoding="utf-8-sig",
    )

    return config, train, output_dir, summary


def test_summary_du_cau_hinh_va_khong_trung(context):
    config, _, _, summary = context

    expected = {
        (embedding, experiment["name"])
        for embedding in config["embedding_artifacts"]
        for experiment in config["hdbscan"]["experiments"]
    }
    actual = set(
        summary[["embedding_run", "experiment"]]
        .itertuples(index=False, name=None)
    )

    assert actual == expected
    assert len(summary) == len(expected)
    assert not summary.duplicated(
        ["embedding_run", "experiment"]
    ).any()

    # Hash trong summary phải khớp cấu hình M10 và M11 hiện tại.
    with (ROOT / "configs/embeddings.yaml").open(
        encoding="utf-8"
    ) as file:
        embedding_config = yaml.safe_load(file)

    expected_hash = get_config_hash(config, embedding_config)

    assert "config_hash" in summary.columns
    assert summary["config_hash"].notna().all()
    assert summary["config_hash"].eq(expected_hash).all()


@pytest.mark.parametrize("embedding_name", EMBEDDING_NAMES)
def test_assignments_va_summary_khop_nhau(context, embedding_name):
    config, train, output_dir, summary = context

    table = pd.read_csv(
        output_dir / f"{embedding_name}_assignments.csv",
        encoding="utf-8-sig",
        dtype={"article_id": str},
    )

    required = {
        "article_id",
        "cluster_id",
        "membership_strength",
        "model_name",
        "data_version",
        "embedding_run",
        "experiment",
        "config_hash",
    }
    assert required.issubset(table.columns)
    assert table["embedding_run"].eq(embedding_name).all()
    assert not table.duplicated(
        ["experiment", "article_id"]
    ).any()

    experiments = config["hdbscan"]["experiments"]
    assert set(table["experiment"]) == {
        experiment["name"] for experiment in experiments
    }
    assert len(table) == len(train) * len(experiments)

    # Đối chiếu metadata cơ bản và thứ tự ID với artifact M10.
    with np.load(
        ROOT / config["embedding_artifacts"][embedding_name],
        allow_pickle=False,
    ) as artifact:
        artifact_ids = artifact["article_ids"].tolist()
        model_name = str(artifact["model_name"][0])
        data_version = str(artifact["data_version"][0])

    assert artifact_ids == train["article_id"].tolist()
    assert table["model_name"].eq(model_name).all()
    assert table["data_version"].eq(data_version).all()

    for experiment in experiments:
        name = experiment["name"]
        group = table.loc[table["experiment"] == name]

        assert group["article_id"].tolist() == artifact_ids

        labels = group["cluster_id"].to_numpy(dtype=float)
        assert np.isfinite(labels).all()
        assert np.all(labels >= -1)
        assert np.all(labels == np.floor(labels))

        strengths = group["membership_strength"].to_numpy(dtype=float)
        assert np.isfinite(strengths).all()
        assert np.all((strengths >= 0) & (strengths <= 1))

        rows = summary.loc[
            (summary["embedding_run"] == embedding_name)
            & (summary["experiment"] == name)
        ]
        assert len(rows) == 1
        row = rows.iloc[0]

        # Assignments và summary của cùng thí nghiệm phải cùng hash.
        assert group["config_hash"].eq(row["config_hash"]).all()

        sizes = group.loc[
            group["cluster_id"] != -1, "cluster_id"
        ].value_counts()
        n_noise = int((group["cluster_id"] == -1).sum())

        assert row["n_articles"] == len(group)
        assert row["n_clusters"] == len(sizes)
        assert row["n_noise"] == n_noise
        assert row["noise_percent"] == pytest.approx(
            round(100 * n_noise / len(group), 2)
        )
        assert row["smallest_cluster"] == (
            int(sizes.min()) if len(sizes) else 0
        )
        assert row["largest_cluster"] == (
            int(sizes.max()) if len(sizes) else 0
        )

        for field in (
            "min_cluster_size",
            "min_samples",
            "cluster_selection_method",
        ):
            assert row[field] == experiment[field]