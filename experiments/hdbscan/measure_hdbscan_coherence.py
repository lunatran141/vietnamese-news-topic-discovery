"""Tính document-level NPMI cho 101 topic của run HDBSCAN đã chọn.

Đây là phép đo mô tả sau huấn luyện. Từ khóa trong common artifact
đã được rút từ train + val + test; không gọi kết quả là test coherence.
Script chỉ đọc artifact đã có, không fit lại HDBSCAN.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.nmf.metrics import npmi_coherence  # noqa: E402

RUN_ID = "minilm_title_content_eom_15_5"
COMMON = ROOT / "artifacts" / "common" / "hdbscan" / RUN_ID
ARTICLES = ROOT / "data" / "processed" / "p0_minimal.csv"


def main() -> None:
    model_summary = json.loads(
        (COMMON / "model_summary.json").read_text(encoding="utf-8-sig")
    )
    if model_summary["run_id"] != RUN_ID:
        raise ValueError("model_summary.json không đúng run_id")

    documents = pd.read_csv(
        COMMON / "document_topics.csv",
        usecols=["article_id", "topic_id", "split"],
        dtype={"article_id": str},
    )
    keywords = pd.read_csv(
        COMMON / "topic_keywords.csv",
        usecols=["topic_id", "rank", "keyword"],
    ).sort_values(["topic_id", "rank"])
    articles = pd.read_csv(
        ARTICLES,
        usecols=["article_id", "processed_title", "processed_content"],
        dtype={"article_id": str},
    )

    topic_ids = sorted(set(documents["topic_id"]) - {-1})
    if len(topic_ids) != 101:
        raise ValueError(f"Cần 101 topic, hiện có {len(topic_ids)}")
    if set(keywords["topic_id"]) != set(topic_ids):
        raise ValueError("Topic ID trong topic_keywords.csv không khớp")
    if len(keywords) != 1010:
        raise ValueError("Cần đúng 10 từ khóa cho mỗi topic")
    if documents["article_id"].duplicated().any():
        raise ValueError("document_topics.csv có article_id trùng")
    if articles["article_id"].duplicated().any():
        raise ValueError("p0_minimal.csv có article_id trùng")

    merged = documents.merge(
        articles,
        on="article_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Có bài trong document_topics.csv thiếu văn bản P0")

    titles = merged["processed_title"].fillna("").astype(str).str.strip()
    contents = (
        merged["processed_content"]
        .fillna("")
        .astype(str)
        .map(lambda value: " ".join(value.split()[:60]))
    )
    texts = (titles + ". " + contents).str.strip(" .")

    # Dùng cùng kiểu unigram/bigram mặc định của TfidfVectorizer
    # trong experiments/hdbscan/export_p01_hdbscan.py.
    terms = list(dict.fromkeys(keywords["keyword"].astype(str)))
    vectorizer = CountVectorizer(
        vocabulary=terms,
        ngram_range=(1, 2),
        binary=True,
    )
    document_term_matrix = vectorizer.transform(texts)

    frequencies = np.asarray(document_term_matrix.sum(axis=0)).ravel()
    missing_terms = [
        term for term, frequency in zip(terms, frequencies) if frequency == 0
    ]
    if missing_terms:
        raise ValueError(
            f"Có {len(missing_terms)} từ khóa không xuất hiện trong corpus: "
            f"{missing_terms[:10]}"
        )

    topic_feature_ids = []
    for topic_id, group in keywords.groupby("topic_id", sort=True):
        if group["rank"].tolist() != list(range(1, 11)):
            raise ValueError(f"Topic {topic_id} không đủ rank 1–10")
        topic_feature_ids.append(
            [vectorizer.vocabulary_[word] for word in group["keyword"]]
        )

    scores_by_split = {}
    topic_rows = []

    for split_name in ["train", "val", "test", "all"]:
        if split_name == "all":
            split_matrix = document_term_matrix
        else:
            mask = merged["split"].eq(split_name).to_numpy()
            split_matrix = document_term_matrix[mask]

        if split_matrix.shape[0] == 0:
            raise ValueError(f"Split {split_name} không có bài")

        score, per_topic_scores = npmi_coherence(
            split_matrix,
            topic_feature_ids,
        )

        split_frequencies = np.asarray(
            split_matrix.sum(axis=0)
        ).ravel()
        missing_ids = set(np.flatnonzero(split_frequencies == 0))
        topics_with_all_keywords = 0

        for topic_id, feature_ids, topic_score in zip(
            topic_ids, topic_feature_ids, per_topic_scores
        ):
            missing_in_topic = sum(
                feature_id in missing_ids for feature_id in feature_ids
            )
            if missing_in_topic == 0:
                topics_with_all_keywords += 1

            topic_rows.append({
                "split": split_name,
                "topic_id": topic_id,
                "npmi_coherence": topic_score,
                "missing_keyword_count": missing_in_topic,
            })

        scores_by_split[split_name] = {
            "npmi_coherence": score,
            "reference_document_count": int(split_matrix.shape[0]),
            "missing_keyword_count": len(missing_ids),
            "topics_with_all_keywords_present": topics_with_all_keywords,
            "topic_count": len(topic_ids),
        }

    summary = {
        "run_id": RUN_ID,
        "metric": "document_npmi",
        "top_n_keywords": 10,
        "topic_count": len(topic_ids),
        "reference_text": (
            "processed_title + 60 từ đầu processed_content"
        ),
        "keyword_source": "topic_keywords.csv; train + val + test",
        "missing_pair_policy": (
            "Cặp từ không đồng xuất hiện nhận điểm -1 theo "
            "experiments.nmf.metrics.npmi_coherence"
        ),
        "outlier_topic_excluded": True,
        "interpretation": (
            "Điểm theo từng tập văn bản, dùng chung bộ từ khóa được rút "
            "từ train + val + test; không phải phép đánh giá test độc lập."
        ),
        "scores_by_split": scores_by_split,
        # Giữ khóa cũ để dashboard HDBSCAN hiện tại tiếp tục đọc điểm all.
        "npmi_coherence": scores_by_split["all"]["npmi_coherence"],
    }

    (COMMON / "coherence_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(topic_rows).to_csv(
        COMMON / "topic_coherence.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for split_name, result in scores_by_split.items():
        print(
            f"{split_name}: NPMI={result['npmi_coherence']:.4f}, "
            f"bài={result['reference_document_count']}, "
            f"từ khóa vắng mặt={result['missing_keyword_count']}, "
            f"cụm có đủ từ khóa="
            f"{result['topics_with_all_keywords_present']}/101"
        )
    print(COMMON / "coherence_summary.json")
    print(COMMON / "topic_coherence.csv")
if __name__ == "__main__":
    main()