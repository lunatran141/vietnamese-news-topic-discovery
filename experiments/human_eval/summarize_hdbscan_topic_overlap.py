from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/hdbscan/human_eval"
OUTPUT = BASE / "summary/overlap_agreement.csv"

REQUIRED = {
    "reviewer_id",
    "blind_topic_id",
    "overlaps_other_topic",
    "overlap_topic_ids",
    "overlap_reason",
}


def load_form(reviewer: str, expected_ids: set[str]) -> pd.DataFrame:
    path = BASE / f"topic_overlap_{reviewer}.csv"
    frame = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} thiếu cột: {sorted(missing)}")

    for column in REQUIRED:
        frame[column] = frame[column].str.strip()

    if not frame["reviewer_id"].eq(reviewer).all():
        raise ValueError(f"{path.name}: reviewer_id không đúng")

    if frame["blind_topic_id"].duplicated().any():
        raise ValueError(f"{path.name}: blind_topic_id bị trùng")

    if set(frame["blind_topic_id"]) != expected_ids:
        raise ValueError(f"{path.name}: thiếu hoặc thừa topic")

    if not frame["overlaps_other_topic"].isin({"yes", "no"}).all():
        raise ValueError(
            f"{path.name}: overlaps_other_topic chỉ nhận yes/no"
        )

    yes = frame["overlaps_other_topic"].eq("yes")
    if frame.loc[yes, "overlap_topic_ids"].eq("").any():
        raise ValueError(
            f"{path.name}: dòng yes phải ghi overlap_topic_ids"
        )
    if frame.loc[~yes, "overlap_topic_ids"].ne("").any():
        raise ValueError(
            f"{path.name}: dòng no phải để trống overlap_topic_ids"
        )
    if frame["overlap_reason"].eq("").any():
        raise ValueError(f"{path.name}: thiếu overlap_reason")

    return frame[
        ["blind_topic_id", "overlaps_other_topic",
         "overlap_topic_ids", "overlap_reason"]
    ]


def main() -> None:
    mapping = pd.read_csv(
        BASE / "topic_mapping.csv",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    expected_ids = set(mapping["blind_topic_id"])

    r1 = load_form("R1", expected_ids)
    r2 = load_form("R2", expected_ids)

    result = r1.merge(
        r2,
        on="blind_topic_id",
        suffixes=("_R1", "_R2"),
        validate="one_to_one",
    )
    result = result.merge(
        mapping[["blind_topic_id", "cluster_id"]],
        on="blind_topic_id",
        validate="one_to_one",
    )
    result["same_decision"] = (
        result["overlaps_other_topic_R1"]
        == result["overlaps_other_topic_R2"]
    )

    if OUTPUT.exists():
        raise FileExistsError(
            f"{OUTPUT} đã tồn tại; kiểm tra trước khi tạo lại"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    total = len(result)
    agreed = int(result["same_decision"].sum())
    print(f"Số topic: {total}")
    print(f"R1/R2 cùng kết luận: {agreed}/{total}")
    print(f"Tỷ lệ trùng kết luận: {agreed / total:.0%}")
    print(f"Kết quả: {OUTPUT}")


if __name__ == "__main__":
    main()