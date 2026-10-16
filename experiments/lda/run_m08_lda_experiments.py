
"""M08: LDA experiments with different topic numbers."""

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.dataset_loader import load_split_dataset
from src.features.article_text import build_article_text
from src.features.bow import (
    fit_bow_pipeline,
    transform_bow_pipeline,
)
from src.models.lda_pipeline import (
    fit_lda_pipeline,
    transform_lda_pipeline,
)
from src.evaluation.lda_metrics import (
    extract_topic_words,
    topic_diversity,
    prepare_coherence_corpus,
    calculate_coherence,
)


def main():
    # 1. Load configurations
    with open(
        "configs/lda_experiments.yaml",
        encoding="utf-8",
    ) as f:
        cfg = yaml.safe_load(f)

    with open(cfg["base_config"], encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    data_cfg = cfg["dataset"]

    output_dir = Path(cfg["outputs"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load train and validation
    train_df = load_split_dataset(
        data_cfg["path"],
        data_cfg["split_manifest_path"],
        data_cfg["train_split"],
    )

    val_df = load_split_dataset(
        data_cfg["path"],
        data_cfg["split_manifest_path"],
        data_cfg["val_split"],
    )

    train_ids = set(train_df["article_id"].astype(str))
    val_ids = set(val_df["article_id"].astype(str))

    if train_ids & val_ids:
        raise ValueError("Train/Validation leakage detected")

    # 3. Prepare text
    train_texts = build_article_text(
        train_df, data_cfg["text_variant"]
    ).tolist()

    val_texts = build_article_text(
        val_df, data_cfg["text_variant"]
    ).tolist()

    print("Train documents:", len(train_texts))
    print("Validation documents:", len(val_texts))

    # 4. Fit BoW only on Train
    vectorizer, train_bow = fit_bow_pipeline(
        train_texts,
        base_cfg["vectorizer"],
    )

    val_bow = transform_bow_pipeline(
        vectorizer,
        val_texts,
    )

    print("Vocabulary size:", train_bow.shape[1])

    # 5. Coherence corpus from Validation
    coherence_texts, dictionary = prepare_coherence_corpus(
        val_texts,
        vectorizer,
    )

    feature_names = vectorizer.get_feature_names_out()

    results = []
    keyword_rows = []
    representative_rows = []

    # 6. Train and evaluate each K
    for k in cfg["experiments"]["topic_numbers"]:
        print(f"\nTraining LDA with K={k}...", flush=True)

        start = time.perf_counter()

        lda_cfg = copy.deepcopy(base_cfg["lda"])
        lda_cfg["n_components"] = int(k)
        lda_cfg["random_state"] = cfg["experiments"][
            "random_state"
        ]

        model = fit_lda_pipeline(train_bow, lda_cfg)

        val_topics = transform_lda_pipeline(
            model,
            val_bow,
        )

        topics = extract_topic_words(
            model,
            feature_names,
            top_n=cfg["experiments"]["top_n_words"],
        )

        cv = calculate_coherence(
            topics,
            coherence_texts,
            dictionary,
            metric="c_v",
            processes=1,
        )

        npmi = calculate_coherence(
            topics,
            coherence_texts,
            dictionary,
            metric="c_npmi",
            processes=1,
        )

        diversity = topic_diversity(topics)

        runtime = round(
            time.perf_counter() - start, 2
        )

        # 7. Save metrics
        results.append({
            "k": k,
            "coherence_cv": cv,
            "coherence_npmi": npmi,
            "topic_diversity": diversity,
            "runtime_seconds": runtime,
        })

        # 8. Save keywords and representative articles
        for topic_id, words in enumerate(topics):
            keyword_rows.append({
                "k": k,
                "topic_id": topic_id,
                "keywords": ", ".join(words),
            })

            best_idx = int(
                np.argmax(val_topics[:, topic_id])
            )

            representative_rows.append({
                "k": k,
                "topic_id": topic_id,
                "article_id": str(
                    val_df.iloc[best_idx]["article_id"]
                ),
                "topic_probability": float(
                    val_topics[best_idx, topic_id]
                ),
                "title": str(
                    val_df.iloc[best_idx]["title"]
                ),
            })

        # 9. Save intermediate results after every K
        pd.DataFrame(results).to_csv(
            output_dir / "topic_number_results.csv",
            index=False,
            encoding="utf-8-sig",
        )

        pd.DataFrame(keyword_rows).to_csv(
            output_dir / "topic_keywords.csv",
            index=False,
            encoding="utf-8-sig",
        )

        pd.DataFrame(representative_rows).to_csv(
            output_dir / "representative_docs.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"K={k} | c_v={cv:.4f} | "
            f"c_npmi={npmi:.4f} | "
            f"diversity={diversity:.4f} | "
            f"runtime={runtime:.2f}s",
            flush=True,
        )

    print("\nM08 experiments completed.")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
