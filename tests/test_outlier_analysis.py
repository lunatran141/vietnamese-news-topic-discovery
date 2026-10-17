from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "hdbscan" / "outlier_analysis"
KEY = ["embedding_run", "experiment", "article_id"]


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUT / name, encoding="utf-8-sig")


def row_keys(table: pd.DataFrame) -> set[tuple]:
    return set(table[KEY].itertuples(index=False, name=None))


def test_outlier_rates_khop_m11() -> None:
    rates = read_csv("m12_outlier_rates.csv")
    summary = pd.read_csv(ROOT / "results/hdbscan/m11_summary.csv")

    overall = rates.loc[rates["group_type"] == "overall"]
    assert len(overall) == len(summary) == 32

    checked = overall.merge(
        summary[
            ["embedding_run", "experiment", "n_articles", "n_noise"]
        ],
        on=["embedding_run", "experiment"],
        validate="one_to_one",
        suffixes=("_m12", "_m11"),
    )

    assert len(checked) == 32
    assert checked["n_articles_m12"].eq(
        checked["n_articles_m11"]
    ).all()
    assert checked["n_outliers"].eq(checked["n_noise"]).all()

    expected_percent = (
        100 * checked["n_outliers"] / checked["n_articles_m12"]
    ).round(2)
    assert checked["outlier_percent"].eq(expected_percent).all()


def test_bai_da_gan_nhan_la_outlier_dung_run() -> None:
    candidates = read_csv("m12_candidates.csv")
    labeled = read_csv("m12_labeled_cases.csv")

    assert len(candidates) == len(labeled) == 60
    assert not labeled.duplicated(KEY).any()
    assert row_keys(labeled) == row_keys(candidates)
    assert labeled.groupby("experiment").size().to_dict() == {
        "eom_15_5": 30,
        "leaf_15_5": 30,
    }

    allowed_labels = {
        "data_noise",
        "rare_valid",
        "multi_topic",
        "valid_outlier",
        "uncertain",
    }
    assert set(labeled["review_label"]).issubset(allowed_labels)

    for column in ["review_label", "evidence", "review_reason"]:
        assert labeled[column].fillna("").astype(str).str.strip().ne(
            ""
        ).all()

    assignments = pd.read_csv(
        ROOT / "results/hdbscan/mpnet_title_content_assignments.csv"
    )
    selected = assignments.loc[
        assignments["experiment"].isin(["eom_15_5", "leaf_15_5"]),
        KEY + ["cluster_id", "config_hash"],
    ]

    checked = labeled.merge(
        selected,
        on=KEY,
        how="left",
        validate="one_to_one",
        suffixes=("_label", "_m11"),
        indicator=True,
    )

    assert checked["_merge"].eq("both").all()
    assert checked["cluster_id_label"].eq(-1).all()
    assert checked["cluster_id_m11"].eq(-1).all()
    assert checked["config_hash_label"].eq(
        checked["config_hash_m11"]
    ).all()


def test_ba_hang_xom_khop_file_nhan() -> None:
    labeled = read_csv("m12_labeled_cases.csv")
    neighbors = read_csv("m12_candidate_neighbors.csv")

    assert len(neighbors) == 180

    for row in labeled.itertuples(index=False):
        found = neighbors.loc[
            neighbors["embedding_run"].eq(row.embedding_run)
            & neighbors["experiment"].eq(row.experiment)
            & neighbors["outlier_article_id"].eq(row.article_id)
        ].sort_values("neighbor_rank")

        assert found["neighbor_rank"].tolist() == [1, 2, 3]
        saved_ids = [
            item.strip()
            for item in row.neighbor_article_ids.split(";")
        ]
        assert saved_ids == found["neighbor_article_id"].tolist()