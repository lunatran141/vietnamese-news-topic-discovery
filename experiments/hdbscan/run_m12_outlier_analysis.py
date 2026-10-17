"""Chạy phân tích tỷ lệ và lấy mẫu outlier cho Task M12."""

from pathlib import Path

import yaml

from src.evaluation.outlier_analysis import (
    calculate_outlier_rates,
    find_nearest_cluster_articles,
    load_m12_inputs,
    sample_outlier_candidates,
)


CONFIG_PATH = Path("configs/outlier_analysis.yaml")


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    train, summary, assignments = load_m12_inputs(config)
    rates = calculate_outlier_rates(train, assignments)
    candidates = sample_outlier_candidates(
        train,
        assignments,
        config,
    )
    neighbors = find_nearest_cluster_articles(
        candidates,
        assignments,
        train,
        config["case_review"]["embedding_path"],
        top_k=3,
    )

    # Đối chiếu tỷ lệ toàn bộ run với summary M11.
    overall = rates.loc[
        rates["group_type"] == "overall",
        [
            "embedding_run",
            "experiment",
            "n_articles",
            "n_outliers",
        ],
    ]
    checked = overall.merge(
        summary[
            ["embedding_run", "experiment", "n_articles", "n_noise"]
        ],
        on=["embedding_run", "experiment"],
        how="inner",
        validate="one_to_one",
        suffixes=("_m12", "_m11"),
    )

    if len(checked) != len(summary):
        raise ValueError("M12 chưa tính đủ các run trong summary M11")

    if not checked["n_articles_m12"].eq(
        checked["n_articles_m11"]
    ).all():
        raise ValueError("Số bài M12 không khớp M11")

    if not checked["n_outliers"].eq(checked["n_noise"]).all():
        raise ValueError("Số outlier M12 không khớp M11")

    if candidates.duplicated(
        ["embedding_run", "experiment", "article_id"]
    ).any():
        raise ValueError("Bảng mẫu có article_id bị trùng trong một run")

    if len(neighbors) != len(candidates) * 3:
        raise ValueError("Mỗi bài mẫu phải có ba bài gần nhất")

    if neighbors.duplicated(
        [
            "embedding_run",
            "experiment",
            "outlier_article_id",
            "neighbor_rank",
        ]
    ).any():
        raise ValueError("Bảng bài gần nhất có dòng bị trùng")

    output_dir = Path(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    rates_path = output_dir / "m12_outlier_rates.csv"
    candidates_path = output_dir / "m12_candidates.csv"
    neighbors_path = output_dir / "m12_candidate_neighbors.csv"

    rates.to_csv(rates_path, index=False, encoding="utf-8-sig")
    candidates.to_csv(
        candidates_path,
        index=False,
        encoding="utf-8-sig",
    )
    neighbors.to_csv(
        neighbors_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Số run đã kiểm tra: {len(overall)}")
    print(f"Số bài mẫu: {len(candidates)}")
    print(f"Số bài gần nhất: {len(neighbors)}")
    print(f"Đã lưu: {rates_path}")
    print(f"Đã lưu: {candidates_path}")
    print(f"Đã lưu: {neighbors_path}")


if __name__ == "__main__":
    main()