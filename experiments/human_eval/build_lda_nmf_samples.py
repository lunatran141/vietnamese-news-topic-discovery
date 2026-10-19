
"""Build reproducible blind human-evaluation samples for LDA and NMF."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
N_TOP = 3
N_RANDOM = 2

LDA_DIR = Path("artifacts/common/lda/lda_k15_seed42")
NMF_DIR = Path("artifacts/nmf/nmf_p3_k10_seed42_v1")

DATASET_PATH = Path("data/processed/p2_segmented.csv")
SPLIT_PATH = Path("data/splits/split_manifest.csv")

OUTPUT_DIR = Path("results/human_eval/lda_nmf")


def load_inputs():
    """Load and validate frozen final-model artifacts."""

    dataset = pd.read_csv(DATASET_PATH)
    manifest = pd.read_csv(SPLIT_PATH)

    lda_docs = pd.read_csv(LDA_DIR / "document_topics.csv")
    lda_reps = pd.read_csv(LDA_DIR / "representative_docs.csv")
    lda_keywords = pd.read_csv(LDA_DIR / "topic_keywords.csv")

    nmf_docs = pd.read_parquet(
        NMF_DIR / "development_document_topics.parquet"
    )
    nmf_keywords = json.loads(
        (NMF_DIR / "topic_keywords.json").read_text(encoding="utf-8")
    )

    assert len(dataset) == 9274
    assert len(manifest) == 9274
    assert len(lda_docs) == 9274
    assert len(lda_reps) == 45
    assert len(lda_keywords) == 150
    assert len(nmf_docs) == 7884
    assert len(nmf_keywords) == 10

    development_ids = set(
        manifest.loc[
            manifest["split"].isin(["train", "val"]), "article_id"
        ].astype(str)
    )

    assert len(development_ids) == 7884

    lda_dev = lda_docs[
        lda_docs["split"].isin(["train", "val"])
    ].copy()

    assert len(lda_dev) == 7884
    assert set(lda_dev["article_id"].astype(str)) == development_ids
    assert set(nmf_docs["article_id"].astype(str)) == development_ids

    assert set(lda_reps["article_id"].astype(str)).issubset(
        development_ids
    )

    print("=== HUMAN EVALUATION INPUT VALIDATION ===")
    print(f"Dataset: {len(dataset)}")
    print(f"Development: {len(development_ids)}")
    print(f"LDA topics: {lda_keywords['topic_id'].nunique()}")
    print(f"LDA representatives: {len(lda_reps)}")
    print(f"NMF topics: {len(nmf_keywords)}")
    print("Input validation: PASS")

    return {
        "dataset": dataset,
        "manifest": manifest,
        "lda_docs": lda_dev,
        "lda_reps": lda_reps,
        "lda_keywords": lda_keywords,
        "nmf_docs": nmf_docs,
        "nmf_keywords": nmf_keywords,
    }



def select_samples(inputs):
    """Select 3 top-ranked and 2 random documents per topic."""

    rng = np.random.default_rng(SEED)
    selected = []

    # 1. LDA: use validated Final representative documents.
    lda_docs = inputs["lda_docs"].copy()
    lda_reps = inputs["lda_reps"].copy()

    for topic_id in range(15):
        topic_docs = lda_docs[
            lda_docs["topic_id"] == topic_id
        ].copy()

        top = lda_reps[
            lda_reps["topic_id"] == topic_id
        ].sort_values("rank")

        assert len(top) == N_TOP
        assert top["rank"].tolist() == [1, 2, 3]

        used_ids = set(top["article_id"].astype(str))

        for _, doc in top.iterrows():
            selected.append({
                "model": "LDA",
                "topic_id": str(topic_id),
                "article_id": str(doc["article_id"]),
                "sample_type": "top",
                "sample_rank": int(doc["rank"]),
            })

        candidates = topic_docs[
            ~topic_docs["article_id"].astype(str).isin(used_ids)
        ].sort_values("article_id")

        assert len(candidates) >= N_RANDOM

        chosen = rng.choice(
            candidates["article_id"].astype(str).to_numpy(),
            size=N_RANDOM,
            replace=False,
        )

        for rank, article_id in enumerate(chosen, start=1):
            selected.append({
                "model": "LDA",
                "topic_id": str(topic_id),
                "article_id": str(article_id),
                "sample_type": "random",
                "sample_rank": rank,
            })

    # 2. NMF: derive dominant topics from Development weights.
    nmf_docs = inputs["nmf_docs"].copy()
    topic_cols = [f"topic_{i:02d}_share" for i in range(10)]

    weights = nmf_docs[topic_cols].to_numpy(dtype=float)

    assert np.isfinite(weights).all()

    nmf_docs["dominant_topic"] = weights.argmax(axis=1)

    for topic_id in range(10):
        topic_docs = nmf_docs[
            nmf_docs["dominant_topic"] == topic_id
        ].copy()

        score_col = f"topic_{topic_id:02d}_share"

        topic_docs = topic_docs.sort_values(
            [score_col, "article_id"],
            ascending=[False, True],
        )

        assert len(topic_docs) >= N_TOP + N_RANDOM

        top = topic_docs.head(N_TOP)
        candidates = topic_docs.iloc[N_TOP:].sort_values("article_id")

        for rank, (_, doc) in enumerate(top.iterrows(), start=1):
            selected.append({
                "model": "NMF",
                "topic_id": f"NMF_{topic_id:02d}",
                "article_id": str(doc["article_id"]),
                "sample_type": "top",
                "sample_rank": rank,
            })

        chosen = rng.choice(
            candidates["article_id"].astype(str).to_numpy(),
            size=N_RANDOM,
            replace=False,
        )

        for rank, article_id in enumerate(chosen, start=1):
            selected.append({
                "model": "NMF",
                "topic_id": f"NMF_{topic_id:02d}",
                "article_id": str(article_id),
                "sample_type": "random",
                "sample_rank": rank,
            })

    samples = pd.DataFrame(selected)

    assert len(samples) == 125
    assert samples.groupby(["model", "topic_id"]).size().eq(5).all()
    assert not samples.duplicated(
        ["model", "topic_id", "article_id"]
    ).any()

    development_ids = set(
        inputs["manifest"].loc[
            inputs["manifest"]["split"].isin(["train", "val"]),
            "article_id",
        ].astype(str)
    )

    assert samples["article_id"].isin(development_ids).all()

    print("\n=== HUMAN EVALUATION SAMPLING ===")
    print(samples.groupby(["model", "sample_type"]).size())
    print("Total samples:", len(samples))
    print("Test leakage: NONE")
    print("Sampling validation: PASS")

    return samples


def build_evaluation_data(inputs, samples):
    """Attach keywords and article text; blind model identities."""

    # Shared readable text for both LDA and NMF.
    articles = inputs["dataset"][
        ["article_id", "title", "content"]
    ].copy()
    articles["article_id"] = articles["article_id"].astype(str)

    if articles["article_id"].duplicated().any():
        raise ValueError("Duplicate article IDs in dataset")

    # LDA: top-10 keywords per topic.
    lda_kw = inputs["lda_keywords"].sort_values(
        ["topic_id", "rank"]
    )

    keyword_lookup = {}

    for topic_id, group in lda_kw.groupby("topic_id"):
        keyword_lookup[("LDA", str(topic_id))] = ", ".join(
            group["keyword"].astype(str).tolist()
        )

    # NMF: top-10 keywords per topic.
    for item in inputs["nmf_keywords"]:
        keyword_lookup[("NMF", item["topic_id"])] = ", ".join(
            entry["term"] for entry in item["keywords"][:10]
        )

    # Shuffle topic order, without revealing model names.
    topic_keys = sorted(
        set(zip(samples["model"], samples["topic_id"]))
    )

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(topic_keys))

    topic_mapping = pd.DataFrame([
        {
            "blind_topic_id": f"T{i + 1:02d}",
            "model": topic_keys[int(position)][0],
            "original_topic_id": topic_keys[int(position)][1],
        }
        for i, position in enumerate(order)
    ])

    merged = samples.merge(
        topic_mapping,
        left_on=["model", "topic_id"],
        right_on=["model", "original_topic_id"],
        how="left",
        validate="many_to_one",
    )

    merged = merged.merge(
        articles,
        on="article_id",
        how="left",
        validate="many_to_one",
    )

    if merged[["title", "content", "blind_topic_id"]].isna().any().any():
        raise ValueError("Missing article content or blind topic ID")

    merged["keywords"] = [
        keyword_lookup[(model, topic)]
        for model, topic in zip(merged["model"], merged["topic_id"])
    ]

    # Remove original model and topic IDs from reviewer-facing data.
    reviewer_data = merged[
        [
            "blind_topic_id",
            "article_id",
            "keywords",
            "title",
            "content",
            "sample_type",
            "sample_rank",
        ]
    ].copy()

    reviewer_data = reviewer_data.sort_values(
        ["blind_topic_id", "sample_type", "sample_rank"]
    ).reset_index(drop=True)

    assert len(reviewer_data) == 125
    assert reviewer_data["blind_topic_id"].nunique() == 25
    assert reviewer_data.groupby("blind_topic_id").size().eq(5).all()

    print("\n=== BLIND EVALUATION DATA ===")
    print("Blind topics:", reviewer_data["blind_topic_id"].nunique())
    print("Article-topic pairs:", len(reviewer_data))
    print("Missing content:", reviewer_data["content"].isna().sum())
    print("Blinding validation: PASS")

    return reviewer_data, topic_mapping


def export_evaluation_files(reviewer_data, topic_mapping):
    """Export reproducible samples and blank independent review forms."""

    output_files = [
        "evaluation_samples.csv",
        "topic_mapping.csv",
        "reviewer_R1_topics.csv",
        "reviewer_R1_documents.csv",
        "reviewer_R2_topics.csv",
        "reviewer_R2_documents.csv",
    ]

    existing_files = [
        name for name in output_files
        if (OUTPUT_DIR / name).exists()
    ]

    if existing_files:
        raise FileExistsError(
            "Human Evaluation files already exist. "
            "Refusing to overwrite: "
            + ", ".join(existing_files)
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED + 1)

    # Randomize the presentation order inside each blind topic.
    shuffled_groups = []

    for _, group in reviewer_data.groupby(
        "blind_topic_id", sort=True
    ):
        order = rng.permutation(len(group))
        shuffled_groups.append(group.iloc[order])

    samples = pd.concat(
        shuffled_groups, ignore_index=True
    ).copy()

    samples["sample_id"] = [
        f"S{i:03d}" for i in range(1, len(samples) + 1)
    ]

    # Administrative file: contains sample origin, but not the
    # model identity directly.
    admin_samples = samples[
        [
            "sample_id",
            "blind_topic_id",
            "article_id",
            "sample_type",
            "sample_rank",
            "keywords",
            "title",
            "content",
        ]
    ].copy()

    admin_samples.to_csv(
        OUTPUT_DIR / "evaluation_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    topic_mapping.to_csv(
        OUTPUT_DIR / "topic_mapping.csv",
        index=False,
        encoding="utf-8-sig",
    )


    # Use separate forms for topic-level and document-level scoring.
    # Do not reveal model identity or top/random sample selection.

    topics_form = (
        samples[["blind_topic_id", "keywords"]]
        .drop_duplicates(subset=["blind_topic_id"])
        .sort_values("blind_topic_id")
        .reset_index(drop=True)
    )

    topics_form["keyword_coherence_1_5"] = ""
    topics_form["topic_interpretability_1_5"] = ""
    topics_form["proposed_topic_label"] = ""
    topics_form["mixed_topic_yes_no"] = ""
    topics_form["topic_notes"] = ""

    documents_form = samples[
        [
            "sample_id",
            "blind_topic_id",
            "keywords",
            "title",
            "content",
        ]
    ].copy()

    documents_form["document_relevance_1_5"] = ""
    documents_form["document_notes"] = ""

    for reviewer in ("R1", "R2"):
        reviewer_topics = topics_form.copy()
        reviewer_topics.insert(0, "reviewer_id", reviewer)

        reviewer_documents = documents_form.copy()
        reviewer_documents.insert(0, "reviewer_id", reviewer)

        reviewer_topics.to_csv(
            OUTPUT_DIR / f"reviewer_{reviewer}_topics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        reviewer_documents.to_csv(
            OUTPUT_DIR / f"reviewer_{reviewer}_documents.csv",
            index=False,
            encoding="utf-8-sig",
        )


    assert len(admin_samples) == 125
    assert admin_samples["sample_id"].is_unique
    assert topic_mapping["blind_topic_id"].nunique() == 25

    forbidden = {
        "model", "original_topic_id",
        "sample_type", "sample_rank", "category_gold",
    }

    assert not forbidden.intersection(topics_form.columns)
    assert not forbidden.intersection(documents_form.columns)

    assert len(topics_form) == 25
    assert len(documents_form) == 125
    assert documents_form["sample_id"].is_unique

    print("\n=== HUMAN EVALUATION EXPORT ===")
    print("Samples:", len(admin_samples))
    print("Blind topics:", topic_mapping["blind_topic_id"].nunique())
    print("Reviewer forms: 4 (2 topic forms + 2 document forms)")
    print("Output:", OUTPUT_DIR)
    print("Export validation: PASS")

if __name__ == "__main__":
    inputs = load_inputs()
    samples = select_samples(inputs)

    reviewer_data, topic_mapping = build_evaluation_data(
        inputs, samples
    )

    export_evaluation_files(reviewer_data, topic_mapping)
