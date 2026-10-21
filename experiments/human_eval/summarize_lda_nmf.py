"""Kiểm tra phiếu chấm độc lập, chỉ tổng hợp khi cả hai reviewer chấm đủ."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


TOPIC_SCORES = ["keyword_coherence_1_5", "topic_interpretability_1_5"]
FLAGS = ["mixed_topic_yes_no", "too_broad_yes_no", "hard_to_name_yes_no"]
DOC_SCORE = "document_relevance_1_5"


def summarize(directory: str | Path) -> dict:
    directory = Path(directory)
    samples = pd.read_csv(directory / "evaluation_samples.csv").fillna("")
    mapping = pd.read_csv(directory / "topic_mapping.csv")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if samples["sample_id"].duplicated().any() or mapping["blind_topic_id"].duplicated().any():
        raise ValueError("Bộ mẫu hoặc mapping có ID trùng")
    topics, documents, missing = [], [], 0
    for reviewer in ("R1", "R2"):
        for kind, ids, ratings in (
            ("topics", "blind_topic_id", TOPIC_SCORES + FLAGS),
            ("documents", "sample_id", [DOC_SCORE]),
        ):
            frame = pd.read_csv(directory / f"reviewer_{reviewer}_{kind}.csv").fillna("")
            expected = mapping["blind_topic_id"] if kind == "topics" else samples["sample_id"]
            if frame[ids].duplicated().any() or set(frame[ids]) != set(expected):
                raise ValueError(f"ID thiếu/thừa/trùng trong phiếu {reviewer}/{kind}")
            if not frame["reviewer_id"].eq(reviewer).all():
                raise ValueError("reviewer_id không khớp tên phiếu")
            if kind == "documents":
                original = samples.set_index("sample_id")
                for column in ("blind_topic_id", "keywords", "title", "content"):
                    expected_values = frame["sample_id"].map(original[column])
                    if not frame[column].eq(expected_values).all():
                        raise ValueError(f"Nội dung bộ mẫu đã thay đổi: {column}")
            else:
                original_keywords = samples.drop_duplicates("blind_topic_id").set_index("blind_topic_id")["keywords"]
                if not frame["keywords"].eq(frame["blind_topic_id"].map(original_keywords)).all():
                    raise ValueError("Keywords trong phiếu topic đã thay đổi")
            for column in ratings:
                if column not in frame:
                    raise ValueError(f"Thiếu tiêu chí {column}; tạo bộ phiếu phiên bản mới")
                values = frame[column].astype(str).str.strip()
                missing += int(values.eq("").sum())
                if column in FLAGS:
                    if not values.isin(["", "yes", "no"]).all():
                        raise ValueError(f"{column} chỉ nhận yes/no")
                    frame[column] = values.map({"yes": 1.0, "no": 0.0})
                else:
                    numeric = pd.to_numeric(values.mask(values.eq("")), errors="raise")
                    if not numeric.dropna().isin([1, 2, 3, 4, 5]).all():
                        raise ValueError(f"{column} phải là điểm nguyên 1-5")
                    frame[column] = numeric
            (topics if kind == "topics" else documents).append(frame)
    result = {"status": "pending" if missing else "complete", "missing_ratings": missing,
              "source_runs": {"NMF": metadata["nmf_run_id"], "LDA": metadata["lda_run_id"]},
              "by_model": [], "agreement": []}
    if missing:
        return result
    topic_scores = pd.concat(topics).merge(mapping, on="blind_topic_id", validate="many_to_one")
    document_scores = pd.concat(documents).merge(mapping, on="blind_topic_id", validate="many_to_one")
    # Mỗi topic có cùng trọng số, không coi mọi lượt chấm bài là độc lập.
    doc_topic_means = document_scores.groupby(["model", "reviewer_id", "blind_topic_id"])[DOC_SCORE].mean()
    for model, group in topic_scores.groupby("model"):
        row = {"model": model, "run_id": result["source_runs"][model]}
        row.update({column: float(group[column].mean()) for column in TOPIC_SCORES + FLAGS})
        row[DOC_SCORE] = float(doc_topic_means.loc[model].mean())
        by_type = document_scores.loc[document_scores["model"].eq(model)].merge(
            samples[["sample_id", "sample_type"]], on="sample_id", validate="many_to_one")
        for sample_type, rows in by_type.groupby("sample_type"):
            row[f"{DOC_SCORE}_{sample_type}"] = float(rows.groupby(["reviewer_id", "blind_topic_id"])[DOC_SCORE].mean().mean())
        result["by_model"].append(row)
    for frames, ids, criteria in (
        (topics, "blind_topic_id", TOPIC_SCORES + FLAGS),
        (documents, "sample_id", [DOC_SCORE]),
    ):
        paired = frames[0].merge(frames[1], on=ids, suffixes=("_R1", "_R2"), validate="one_to_one")
        for criterion in criteria:
            left, right = paired[f"{criterion}_R1"], paired[f"{criterion}_R2"]
            # Hằng số chung khiến kappa không xác định; xuất null, không gán 0.
            kappa = None if pd.concat([left, right]).nunique() == 1 else float(
                cohen_kappa_score(left, right, weights=None if criterion in FLAGS else "quadratic"))
            result["agreement"].append({"criterion": criterion,
                "exact_agreement": float(left.eq(right).mean()),
                "mean_absolute_difference": float((left - right).abs().mean()),
                "kappa": kappa if kappa is None or np.isfinite(kappa) else None})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("results/human_eval/lda_nmf"))
    args = parser.parse_args()
    result = summarize(args.directory)
    path = args.directory / "summary.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"{result['status']}: {result['missing_ratings']} missing ratings; {path}")


if __name__ == "__main__":
    main()
