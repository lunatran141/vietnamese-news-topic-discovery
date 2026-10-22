"""Gom các ca HDBSCAN đã phân tích để phục vụ E06."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/hdbscan"
OUTPUT = BASE / "error_analysis/hdbscan_error_cases.csv"

# Các ca validation đã đọc trong đánh giá 4.7.
# Có thể bổ sung ID sau khi đọc và xác nhận lỗi.
HUMAN_CASES = {
    "HS001": "duplicate",
    "HS004": "duplicate",
    "HS013": "broad",
    "HS016": "multi_topic",
    "HS023": "broad",
    "HS024": "broad",
}

# Khóa M12 gồm experiment + article_id vì một bài có thể xuất hiện ở hai run.
OUTLIER_CASES = [
    ("eom_15_5", "TTR_607809186810"),  # valid_outlier
    ("eom_15_5", "TTR_607b52b568c2"),  # rare_valid
    ("eom_15_5", "VNN_14f9af2f4e8b"),  # data_noise
    ("eom_15_5", "VNN_a9503ec90cf8"),  # multi_topic
]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig").fillna("")


def main() -> None:
    samples = read_csv(BASE / "human_eval/evaluation_samples.csv")
    mapping = read_csv(BASE / "human_eval/topic_mapping.csv")
    r1 = read_csv(BASE / "human_eval/reviewer_R1_documents.csv")
    r2 = read_csv(BASE / "human_eval/reviewer_R2_documents.csv")
    outliers = read_csv(BASE / "outlier_analysis/m12_labeled_cases.csv")

    samples = samples.merge(
        mapping[["blind_topic_id", "embedding_run", "experiment"]],
        on="blind_topic_id",
        validate="many_to_one",
    )

    if not samples["sample_id"].is_unique:
        raise ValueError("sample_id trong evaluation_samples.csv không duy nhất")

    samples = samples.set_index("sample_id")
    r1 = r1.set_index("sample_id")
    r2 = r2.set_index("sample_id")
    rows = []

    for sample_id, error_group in HUMAN_CASES.items():
        if sample_id not in samples.index:
            raise ValueError(f"Không tìm thấy {sample_id} trong bộ mẫu")

        item = samples.loc[sample_id]
        reason_r1 = str(r1.loc[sample_id, "review_reason"])
        reason_r2 = str(r2.loc[sample_id, "review_reason"])

        rows.append({
            "case_id": sample_id,
            "split": "val",
            "embedding_run": item["embedding_run"],
            "experiment": item["experiment"],
            "article_id": item["article_id"],
            "cluster_id": item["cluster_id"],
            "error_group": error_group,
            "review_label": "",
            "title": item["title"],
            "evidence": f"R1: {reason_r1} | R2: {reason_r2}",
            "interpretation": "",
            "source_file": "human_eval/evaluation_samples.csv",
        })

    for experiment, article_id in OUTLIER_CASES:
        match = outliers.loc[
            (outliers["experiment"] == experiment)
            & (outliers["article_id"] == article_id)
        ]
        if len(match) != 1:
            raise ValueError(
                f"Cần đúng 1 dòng M12 cho {experiment}, {article_id}; "
                f"tìm thấy {len(match)}"
            )

        item = match.iloc[0]
        rows.append({
            "case_id": f"M12-{experiment}-{article_id}",
            "split": "train",
            "embedding_run": item["embedding_run"],
            "experiment": experiment,
            "article_id": article_id,
            "cluster_id": item["cluster_id"],
            "error_group": "outlier_review",
            "review_label": item["review_label"],
            "title": item["title"],
            "evidence": item["evidence"],
            "interpretation": item["review_reason"],
            "source_file": "outlier_analysis/m12_labeled_cases.csv",
        })
    examples = read_csv(BASE / "topic_naming/topic_name_examples.csv")
    match = examples.loc[
        (examples["article_id"] == "VNN_707b880b93fd")
        & (examples["topic_id"].astype(str) == "1")
        & (examples["split"] == "train")
    ]
    if len(match) != 1:
        raise ValueError(
            "Cần đúng một bài VNN_707b880b93fd thuộc cụm 1 trên train"
        )

    item = match.iloc[0]
    rows.append({
        "case_id": "TN001",
        "split": "train",
        "embedding_run": "minilm_title_content",
        "experiment": "eom_15_5",
        "article_id": item["article_id"],
        "cluster_id": 1,
        "error_group": "broad",
        "review_label": "",
        "title": item["title"],
        "evidence": (
            "Tiêu đề và nội dung nói về trọng lượng laptop "
            "và trải nghiệm mang máy khi di chuyển."
        ),
        "interpretation": (
            "Bài công nghệ bị gán vào cụm y tế T05; "
            "đây là ca ngoài chủ đề được phát hiện khi rà bài train."
        ),
        "source_file": "topic_naming/topic_name_examples.csv",
    })

    result = pd.DataFrame(rows)

    # Bảo vệ kết quả đã rà thủ công khỏi bị ghi đè.
    if OUTPUT.exists():
        raise FileExistsError(f"File đã tồn tại: {OUTPUT}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"Đã lưu {len(result)} ca: {OUTPUT}")


if __name__ == "__main__":
    main()